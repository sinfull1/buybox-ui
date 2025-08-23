import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import * as d3 from 'd3';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    ScatterChart,
    Scatter,
    BarChart,
    Bar,
    Cell
} from 'recharts';


// Custom Hooks for State Management
const useTraining = () => {
    const [isTraining, setIsTraining] = useState(false);
    const [progress, setProgress] = useState(0);
    const [lossHistory, setLossHistory] = useState([]);
    const [currentEpoch, setCurrentEpoch] = useState(0);
    const stopTrainingRef = useRef(false);

    const startTraining = useCallback(async (config, products, salesHistory, trueElasticityMatrix) => {
        setIsTraining(true);
        setProgress(0);
        setLossHistory([]);
        setCurrentEpoch(0);
        stopTrainingRef.current = false;

        let currentElasticityMatrix = MathUtils.generateElasticityMatrix(products, true);
        let finalLoss = Infinity;

        for (let epoch = 0; epoch <= config.epochs; epoch++) {
            if (stopTrainingRef.current) break;

            let totalSquaredError = 0;
            let dataPoints = 0;

            for (const productId in salesHistory) {
                const history = salesHistory[productId];
                for (const point of history) {
                    const productForPrediction = products.find(p => p.id === parseInt(productId));
                    const allProductsWithHistoricalPrices = products.map(p => ({
                        ...p,
                        currentPrice: point.prices[p.id] || p.basePrice
                    }));

                    const predictedDemand = MathUtils.calculateDemand(
                        { ...productForPrediction, currentPrice: point.prices[productId] },
                        allProductsWithHistoricalPrices,
                        currentElasticityMatrix
                    );

                    const error = predictedDemand - point.demand;
                    totalSquaredError += error * error;
                    dataPoints++;
                }
            }

            const mse = totalSquaredError / dataPoints;
            finalLoss = mse;

            setLossHistory(prev => [...prev, { epoch, loss: parseFloat(mse.toFixed(2)) }]);
            setCurrentEpoch(epoch);
            setProgress((epoch / config.epochs) * 100);

            const learningRate = 0.1;
            for (const p1_id in currentElasticityMatrix) {
                for (const p2_id in currentElasticityMatrix[p1_id]) {
                    const currentVal = currentElasticityMatrix[p1_id][p2_id];
                    const trueVal = trueElasticityMatrix[p1_id][p2_id];
                    currentElasticityMatrix[p1_id][p2_id] = currentVal * (1 - learningRate) + trueVal * learningRate;
                }
            }

            await new Promise(resolve => setTimeout(resolve, 30));
        }

        setIsTraining(false);
        return {
            finalLoss,
            trainedMatrix: currentElasticityMatrix,
            status: stopTrainingRef.current ? 'Stopped' : 'Completed'
        };
    }, []);

    const stopTraining = useCallback(() => {
        stopTrainingRef.current = true;
    }, []);

    return { isTraining, progress, lossHistory, currentEpoch, startTraining, stopTraining };
};

const useOptimization = () => {
    const [isOptimizing, setIsOptimizing] = useState(false);
    const [optimizationHistory, setOptimizationHistory] = useState([]);
    const [currentSolution, setCurrentSolution] = useState(null);
    const [convergenceMetrics, setConvergenceMetrics] = useState(null);
    const stopOptimizationRef = useRef(false);

    const startOptimization = useCallback(async (products, elasticityMatrix, constraints, objective = 'profit') => {
        setIsOptimizing(true);
        setOptimizationHistory([]);
        setConvergenceMetrics(null);
        stopOptimizationRef.current = false;

        const { maxPriceChange } = constraints;

        const alpha = 0.05;
        const beta1 = 0.9;
        const beta2 = 0.999;
        const epsilon = 1e-8;

        let currentPrices = products.map(p => p.basePrice);
        let m = new Array(products.length).fill(0);
        let v = new Array(products.length).fill(0);
        let t = 0;

        let bestObjective = -Infinity;
        let bestPrices = [...currentPrices];
        const maxIterations = 200;

        const calculateObjectiveValue = (prices, obj) => {
            const testProducts = products.map((p, i) => ({ ...p, currentPrice: prices[i] }));

            if (obj === 'revenue') {
                return products.reduce((totalRevenue, product, i) => {
                    const demand = MathUtils.calculateDemand({ ...product, currentPrice: prices[i] }, testProducts, elasticityMatrix);
                    return totalRevenue + prices[i] * demand;
                }, 0);
            }

            return products.reduce((totalProfit, product, i) => {
                const demand = MathUtils.calculateDemand({ ...product, currentPrice: prices[i] }, testProducts, elasticityMatrix);
                const profit = (prices[i] - product.cost) * demand;
                return totalProfit + profit;
            }, 0);
        };

        let currentObjectiveValue = calculateObjectiveValue(currentPrices, objective);

        while (t < maxIterations) {
            if (stopOptimizationRef.current) {
                setConvergenceMetrics({ converged: false, iterations: t, reason: 'Stopped by user.' });
                break;
            }
            t++;

            const gradients = [];
            const grad_epsilon = 0.01;

            for (let i = 0; i < products.length; i++) {
                const pricesUp = [...currentPrices];
                pricesUp[i] += grad_epsilon;
                const objectiveUp = calculateObjectiveValue(pricesUp, objective);

                const pricesDown = [...currentPrices];
                pricesDown[i] -= grad_epsilon;
                const objectiveDown = calculateObjectiveValue(pricesDown, objective);

                const gradient = (objectiveUp - objectiveDown) / (2 * grad_epsilon);
                gradients.push(gradient);
            }

            m = m.map((m_i, i) => beta1 * m_i + (1 - beta1) * gradients[i]);
            v = v.map((v_i, i) => beta2 * v_i + (1 - beta2) * Math.pow(gradients[i], 2));

            const m_hat = m.map(m_i => m_i / (1 - Math.pow(beta1, t)));
            const v_hat = v.map(v_i => v_i / (1 - Math.pow(beta2, t)));

            const newPrices = currentPrices.map((price, i) => {
                const update = alpha * m_hat[i] / (Math.sqrt(v_hat[i]) + epsilon);
                let newPrice = price + update;

                const minPrice = products[i].cost * 1.05;
                const maxPriceLimit = products[i].basePrice * (1 + maxPriceChange / 100);
                const minPriceLimit = products[i].basePrice * (1 - maxPriceChange / 100);

                newPrice = Math.max(minPrice, Math.min(maxPriceLimit, Math.max(minPriceLimit, newPrice)));

                return newPrice;
            });

            const newObjectiveValue = calculateObjectiveValue(newPrices, objective);
            const gradientNorm = Math.sqrt(gradients.reduce((sum, g) => sum + g * g, 0));

            setOptimizationHistory(prev => [...prev, {
                iteration: t,
                objective: newObjectiveValue,
                gradient_norm: gradientNorm,
            }]);

            if (newObjectiveValue > bestObjective) {
                bestObjective = newObjectiveValue;
                bestPrices = [...newPrices];

                setCurrentSolution({
                    prices: [...newPrices],
                    expectedRevenue: products.reduce((sum, p, i) => {
                        const demand = MathUtils.calculateDemand({ ...p, currentPrice: newPrices[i] }, products.map((prod, j) => ({ ...prod, currentPrice: newPrices[j] })), elasticityMatrix);
                        return sum + newPrices[i] * demand;
                    }, 0),
                    totalProfit: products.reduce((sum, p, i) => {
                        const demand = MathUtils.calculateDemand({ ...p, currentPrice: newPrices[i] }, products.map((prod, j) => ({ ...prod, currentPrice: newPrices[j] })), elasticityMatrix);
                        return sum + (newPrices[i] - p.cost) * demand;
                    }, 0),
                    priceChanges: newPrices.map((price, i) => (price - products[i].basePrice) / products[i].basePrice)
                });
            }

            if (gradientNorm < 0.5 || Math.abs(newObjectiveValue - currentObjectiveValue) < 1) {
                setConvergenceMetrics({ converged: true, iterations: t, reason: 'Convergence criteria met.' });
                break;
            }

            currentPrices = newPrices;
            currentObjectiveValue = newObjectiveValue;

            await new Promise(resolve => setTimeout(resolve, 30));
        }

        if (t >= maxIterations) {
            setConvergenceMetrics({ converged: false, iterations: t, reason: 'Max iterations reached.' });
        }

        setIsOptimizing(false);
    }, []);

    const stopOptimization = useCallback(() => {
        stopOptimizationRef.current = true;
    }, []);

    const resetOptimization = useCallback(() => {
        setOptimizationHistory([]);
        setCurrentSolution(null);
        setConvergenceMetrics(null);
    }, []);

    return { isOptimizing, optimizationHistory, currentSolution, convergenceMetrics, startOptimization, stopOptimization, resetOptimization };
};


const useNotification = () => {
    const [notifications, setNotifications] = useState([]);

    const addNotification = useCallback((message, type = 'info') => {
        const id = Date.now();
        setNotifications(prev => [...prev, { id, message, type, timestamp: new Date() }]);
        setTimeout(() => {
            setNotifications(prev => prev.filter(n => n.id !== id));
        }, 5000);
    }, []);

    return { notifications, addNotification };
};

const useMonteCarloAnalysis = () => {
    const [isRunning, setIsRunning] = useState(false);
    const [results, setResults] = useState(null);
    const [progress, setProgress] = useState(0);

    const runMonteCarloAnalysis = useCallback(async (products, elasticityMatrix, simulations = 1000) => {
        setIsRunning(true);
        setProgress(0);
        setResults(null);

        const outcomes = [];

        for (let i = 0; i < simulations; i++) {
            const priceVariations = products.map(p => ({
                ...p,
                currentPrice: p.basePrice * (0.8 + Math.random() * 0.4)
            }));

            let totalRevenue = 0;
            let totalProfit = 0;

            priceVariations.forEach(product => {
                const demand = MathUtils.calculateDemand(product, priceVariations, elasticityMatrix);
                const revenue = product.currentPrice * demand;
                const profit = (product.currentPrice - product.cost) * demand;

                totalRevenue += revenue;
                totalProfit += profit;
            });

            outcomes.push({
                simulation: i,
                revenue: totalRevenue,
                profit: totalProfit,
            });

            setProgress((i / simulations) * 100);

            if (i % 50 === 0) {
                await new Promise(resolve => setTimeout(resolve, 1));
            }
        }

        const revenues = outcomes.map(o => o.revenue);
        const profits = outcomes.map(o => o.profit);

        const stats = {
            revenue: {
                mean: revenues.reduce((a, b) => a + b, 0) / revenues.length,
                std: Math.sqrt(revenues.reduce((sq, n) => sq + Math.pow(n - (revenues.reduce((a, b) => a + b, 0) / revenues.length), 2), 0) / revenues.length),
            },
            profit: {
                mean: profits.reduce((a, b) => a + b, 0) / profits.length,
                std: Math.sqrt(profits.reduce((sq, n) => sq + Math.pow(n - (profits.reduce((a, b) => a + b, 0) / profits.length), 2), 0) / profits.length),
            },
            outcomes
        };

        setResults(stats);
        setIsRunning(false);
    }, []);

    return { isRunning, results, progress, runMonteCarloAnalysis };
};

// Mathematical utilities for EvolveGCN
const MathUtils = {
    evolveGCNElasticityExtraction: (products, adjacencyMatrix, randomize = false) => {

        const nodeFeatures = products.map(product => [
            product.basePrice / 100,
            product.baseDemand / 500,
            product.cost / 100,
            product.inventory / 1000,
            product.category === 'Electronics' ? 1 : 0,
            product.category === 'Clothing' ? 1 : 0,
            product.category === 'Books' ? 1 : 0,
            product.category === 'Home & Garden' ? 1 : 0,
            product.category === 'Sports' ? 1 : 0
        ]);

        const performGraphConvolution = (features, adjMatrix, weights) => {
            const result = [];
            for (let i = 0; i < features.length; i++) {
                const nodeResult = new Array(weights[0].length).fill(0);
                for (let j = 0; j < features.length; j++) {
                    const edgeWeight = adjMatrix[i][j];
                    for (let k = 0; k < weights[0].length; k++) {
                        let sum = 0;
                        for (let l = 0; l < features[j].length; l++) {
                            sum += features[j][l] * weights[l][k];
                        }
                        nodeResult[k] += edgeWeight * sum;
                    }
                }
                result.push(nodeResult.map(x => Math.max(0, x)));
            }
            return result;
        };

        const layer1Weights = Array(9).fill().map(() =>
            Array(16).fill().map(() => (Math.random() - 0.5) * 0.1)
        );
        const layer2Weights = Array(16).fill().map(() =>
            Array(8).fill().map(() => (Math.random() - 0.5) * 0.1)
        );

        const layer1Output = performGraphConvolution(nodeFeatures, adjacencyMatrix, layer1Weights);
        const layer2Output = performGraphConvolution(layer1Output, adjacencyMatrix, layer2Weights);

        const elasticityMatrix = {};

        products.forEach((product, i) => {
            elasticityMatrix[product.id] = {};
            products.forEach((otherProduct, j) => {
                let value;
                if (i === j) {
                    const baseElasticity = -0.8 - (product.basePrice / 500);
                    value = baseElasticity - (randomize ? Math.random() * 0.5 : 0);
                } else {
                    const embedding_i = layer2Output[i];
                    const embedding_j = layer2Output[j];

                    let dotProduct = 0;
                    for (let k = 0; k < embedding_i.length; k++) {
                        dotProduct += embedding_i[k] * embedding_j[k];
                    }

                    const magnitude_i = Math.sqrt(embedding_i.reduce((sum, val) => sum + val * val, 0));
                    const magnitude_j = Math.sqrt(embedding_j.reduce((sum, val) => sum + val * val, 0));
                    const similarity = dotProduct / (magnitude_i * magnitude_j + 1e-8);

                    const sameCategory = product.category === otherProduct.category;

                    if (sameCategory) {
                        value = similarity * 0.5;
                    } else {
                        value = similarity * 0.1;
                    }
                    value += (randomize ? (Math.random() - 0.5) * 0.1 : 0);
                }
                elasticityMatrix[product.id][otherProduct.id] = value;
            });
        });

        return elasticityMatrix;
    },

    buildProductGraph: (products) => {
        const adjacencyMatrix = products.map(() => products.map(() => 0));

        for (let i = 0; i < products.length; i++) {
            for (let j = 0; j < products.length; j++) {
                if (i !== j) {
                    const categorySimilarity = products[i].category === products[j].category ? 0.9 : 0.2;
                    const priceDiff = Math.abs(products[i].basePrice - products[j].basePrice);
                    const avgPrice = (products[i].basePrice + products[j].basePrice) / 2;
                    const priceSimilarity = Math.exp(-priceDiff / avgPrice);
                    const demandRatio = Math.min(products[i].baseDemand, products[j].baseDemand) /
                        Math.max(products[i].baseDemand, products[j].baseDemand);

                    adjacencyMatrix[i][j] = 0.4 * categorySimilarity + 0.3 * priceSimilarity + 0.3 * demandRatio;

                    if (adjacencyMatrix[i][j] < 0.15) {
                        adjacencyMatrix[i][j] = 0.05;
                    }
                }
            }
        }

        for (let i = 0; i < products.length; i++) {
            const rowSum = adjacencyMatrix[i].reduce((sum, val) => sum + val, 0);
            if (rowSum > 0) {
                for (let j = 0; j < products.length; j++) {
                    adjacencyMatrix[i][j] = (adjacencyMatrix[i][j] / rowSum) + 0.01;
                }
            }
        }

        return adjacencyMatrix;
    },

    calculateDemand: (product, allProducts, elasticityMatrix) => {
        let demand = product.baseDemand;

        for (let i = 0; i < allProducts.length; i++) {
            const otherProduct = allProducts[i];
            const elasticity = elasticityMatrix[product.id]?.[otherProduct.id];
            if (elasticity === undefined) continue;

            const priceRatio = Math.max(0.1, Math.min(10, otherProduct.currentPrice / otherProduct.basePrice));
            const cappedElasticity = Math.max(-5, Math.min(5, elasticity));

            demand *= Math.pow(priceRatio, cappedElasticity);
        }

        const minDemand = product.baseDemand * 0.01;
        const maxDemand = product.baseDemand * 5;

        return Math.max(minDemand, Math.min(maxDemand, demand));
    },
    generateElasticityMatrix: (products, randomize = false) => {
        const adjacencyMatrix = MathUtils.buildProductGraph(products);
        return MathUtils.evolveGCNElasticityExtraction(products, adjacencyMatrix, randomize);
    }
};

const generateSampleData = () => {
    const categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports', 'Beauty', 'Automotive', 'Health', 'Toys', 'Office'];
    const products = [];

    for (let i = 0; i < 200; i++) {
        const basePrice = 10 + Math.random() * 490;
        const marginPercent = -10 + Math.random() * 35;
        const cost = basePrice * (1 - marginPercent / 100);

        products.push({
            id: i,
            name: `Product ${i + 1}`,
            category: categories[Math.floor(Math.random() * categories.length)],
            basePrice: parseFloat(basePrice.toFixed(2)),
            currentPrice: parseFloat(basePrice.toFixed(2)),
            baseDemand: 50 + Math.random() * 950,
            cost: parseFloat(cost.toFixed(2)),
            inventory: Math.floor(Math.random() * 2000)
        });
    }

    return products;
};

const generateSalesHistory = (products, trueElasticityMatrix) => {
    const history = {};
    const numWeeks = 52;

    products.forEach(p => {
        history[p.id] = [];
    });

    for (let week = 0; week < numWeeks; week++) {
        const weeklyPrices = {};
        products.forEach(p => {
            weeklyPrices[p.id] = p.basePrice * (0.85 + Math.random() * 0.3);
        });

        products.forEach(p => {
            const allProductsWithWeeklyPrices = products.map(prod => ({
                ...prod,
                currentPrice: weeklyPrices[prod.id]
            }));

            const demand = MathUtils.calculateDemand(
                { ...p, currentPrice: weeklyPrices[p.id] },
                allProductsWithWeeklyPrices,
                trueElasticityMatrix
            );

            history[p.id].push({
                week,
                demand: demand * (0.95 + Math.random() * 0.1),
                prices: { ...weeklyPrices }
            });
        });
    }
    return history;
};

const D3Network = ({ products, elasticityMatrix, theme, onNodeClick }) => {
    const svgRef = useRef();
    const [nodes, setNodes] = useState([]);
    const [links, setLinks] = useState([]);

    useEffect(() => {
        if (!products || !elasticityMatrix || Object.keys(elasticityMatrix).length === 0) return;

        const newNodes = products.map(p => ({ ...p }));
        const newLinks = [];
        for (let i = 0; i < products.length; i++) {
            for (let j = i + 1; j < products.length; j++) {
                const p1 = products[i];
                const p2 = products[j];
                const elasticity = elasticityMatrix[p1.id]?.[p2.id] || 0;
                if (Math.abs(elasticity) > 0.1) {
                    newLinks.push({
                        source: p1.id,
                        target: p2.id,
                        value: Math.abs(elasticity),
                        type: elasticity > 0 ? 'substitute' : 'complement'
                    });
                }
            }
        }
        setNodes(newNodes);
        setLinks(newLinks);
    }, [products, elasticityMatrix]);

    useEffect(() => {
        if (nodes.length === 0 || links.length === 0) return;

        const width = 800;
        const height = 600;

        const svg = d3.select(svgRef.current)
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', `0 0 ${width} ${height}`);

        svg.selectAll("*").remove();

        const container = svg.append("g");

        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {
                container.attr("transform", event.transform);
            });
        svg.call(zoom);

        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(d => 150 - d.value * 50))
            .force('charge', d3.forceManyBody().strength(-150))
            .force('center', d3.forceCenter(width / 2, height / 2));

        const link = container.append('g')
            .selectAll('line')
            .data(links)
            .enter().append('line')
            .style('stroke', d => d.type === 'substitute' ? '#ef4444' : '#10b981')
            .style('stroke-width', d => d.value * 3);

        const node = container.append('g')
            .selectAll('circle')
            .data(nodes)
            .enter().append('circle')
            .attr('r', d => 5 + d.baseDemand / 100)
            .style('fill', '#6366f1')
            .style('stroke', theme === 'dark' ? '#fff' : '#333')
            .style('stroke-width', 1.5)
            .style('cursor', 'pointer')
            .on("click", (event, d) => onNodeClick(d))
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        node.append("title").text(d => `${d.name}\nCategory: ${d.category}\nPrice: $${d.basePrice}`);

        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
        });

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

    }, [nodes, links, theme, onNodeClick]);

    return <svg ref={svgRef}></svg>;
};

const MLOpsPlatform = () => {
    const [theme, setTheme] = useState('dark');
    const [activeTab, setActiveTab] = useState('dashboard');
    const [products, setProducts] = useState(generateSampleData());
    const [trueElasticityMatrix, setTrueElasticityMatrix] = useState({});
    const [salesHistory, setSalesHistory] = useState(null);
    const [optimizationConstraints, setOptimizationConstraints] = useState({ maxPriceChange: 30 });
    const [optimizationObjective, setOptimizationObjective] = useState('profit');
    const [trainingConfig, setTrainingConfig] = useState({
        epochs: 100,
        learningRate: 0.001,
        hiddenDim: 64,
        graphMethod: 'hybrid'
    });
    const [selectedNode, setSelectedNode] = useState(null);
    const [selectedModelForAnalysis, setSelectedModelForAnalysis] = useState(null);

    const training = useTraining();
    const optimization = useOptimization();
    const notifications = useNotification();
    const monteCarlo = useMonteCarloAnalysis();

    const [trainingJobs, setTrainingJobs] = useState([]);
    const [modelRegistry, setModelRegistry] = useState([]);
    const [deployments, setDeployments] = useState([]);

    useEffect(() => {
        const adjMatrix = MathUtils.buildProductGraph(products);
        const groundTruth = MathUtils.evolveGCNElasticityExtraction(products, adjMatrix);
        setTrueElasticityMatrix(groundTruth);
    }, [products]);

    const toggleTheme = () => {
        setTheme(prev => prev === 'dark' ? 'light' : 'dark');
    };

    const handleLoadHistory = () => {
        notifications.addNotification('Generating historical sales data...', 'info');
        const history = generateSalesHistory(products, trueElasticityMatrix);
        setSalesHistory(history);
        notifications.addNotification('Sales history loaded and ready for training.', 'success');
    };

    const handleStartTraining = useCallback(async () => {
        if (!salesHistory) {
            notifications.addNotification('Please load sales history before training.', 'error');
            return;
        }

        const jobId = `job-${Date.now()}`;
        const newJob = {
            id: jobId,
            status: 'Running',
            startTime: new Date(),
            duration: 0,
            finalLoss: null
        };
        setTrainingJobs(prev => [newJob, ...prev]);

        const result = await training.startTraining(trainingConfig, products, salesHistory, trueElasticityMatrix);

        setTrainingJobs(prev => prev.map(job => job.id === jobId ? {
            ...job,
            status: result.status,
            duration: (new Date() - job.startTime) / 1000,
            finalLoss: result.finalLoss
        } : job));

        if (result.status === 'Completed') {
            const modelVersion = `v1.0.${modelRegistry.length}`;
            const newModel = {
                version: modelVersion,
                jobId: jobId,
                createdAt: new Date(),
                loss: result.finalLoss,
                elasticityMatrix: result.trainedMatrix
            };
            setModelRegistry(prev => [newModel, ...prev]);
            notifications.addNotification(`New model ${modelVersion} created and added to registry.`, 'success');
        }

    }, [trainingConfig, products, salesHistory, trueElasticityMatrix, training, notifications, modelRegistry.length]);

    const handleDeployModel = (modelVersion) => {
        const newDeployment = {
            id: `deploy-${Date.now()}`,
            modelVersion,
            status: 'Active',
            deployedAt: new Date(),
        };
        const updatedDeployments = deployments.map(d => ({...d, status: 'Inactive'}));
        setDeployments([newDeployment, ...updatedDeployments]);
        notifications.addNotification(`Model ${modelVersion} is now active.`, 'success');
    }

    const handleRunOptimization = () => {
        const model = modelRegistry.find(m => m.version === selectedModelForAnalysis);
        if (!model) {
            notifications.addNotification('Please select a model to run optimization.', 'error');
            return;
        }
        optimization.startOptimization(products, model.elasticityMatrix, optimizationConstraints, optimizationObjective);
        notifications.addNotification(`Optimization started for model ${model.version}.`, 'success');
    };

    const handleRunMonteCarlo = () => {
        const model = modelRegistry.find(m => m.version === selectedModelForAnalysis);
        if (!model) {
            notifications.addNotification('Please select a model to run analysis.', 'error');
            return;
        }
        monteCarlo.runMonteCarloAnalysis(products, model.elasticityMatrix);
        notifications.addNotification(`Monte Carlo analysis started for model ${model.version}.`, 'success');
    };


    const themeClasses = {
        dark: {
            bg: 'bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900',
            cardBg: 'bg-white/10 backdrop-blur-md border border-white/20',
            text: 'text-white',
            textSecondary: 'text-gray-300',
            accent: 'bg-gradient-to-r from-purple-500 to-pink-500'
        },
        light: {
            bg: 'bg-gradient-to-br from-blue-50 via-white to-purple-50',
            cardBg: 'bg-white/70 backdrop-blur-md border border-gray-200',
            text: 'text-gray-900',
            textSecondary: 'text-gray-600',
            accent: 'bg-gradient-to-r from-blue-500 to-purple-500'
        }
    };

    const currentTheme = themeClasses[theme];

    return (
        <div className={`min-h-screen transition-all duration-500 ${currentTheme.bg}`}>
            <header className={`${currentTheme.cardBg} shadow-xl`}>
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center py-6">
                        <div>
                            <h1 className={`text-3xl font-bold ${currentTheme.text}`}>MLOps Platform</h1>
                            <p className={`${currentTheme.textSecondary}`}>For Price Optimization Models</p>
                        </div>
                        <div className="flex items-center space-x-4">
                            <button
                                onClick={toggleTheme}
                                className={`px-4 py-2 rounded-lg ${currentTheme.cardBg} ${currentTheme.text} hover:scale-105 transition-all duration-200`}
                            >
                                {theme === 'dark' ? '☀️' : '🌙'}
                            </button>
                        </div>
                    </div>
                </div>
            </header>

            <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">
                <div className="flex space-x-1 overflow-x-auto">
                    {['dashboard', 'data-management', 'training-jobs', 'model-registry', 'deployments', 'optimization', 'monte-carlo', 'elasticity-matrix', 'monitoring', 'model-assumptions'].map((tab) => (
                        <button
                            key={tab}
                            onClick={() => setActiveTab(tab)}
                            className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 capitalize whitespace-nowrap ${
                                activeTab === tab
                                    ? `${currentTheme.accent} text-white shadow-lg`
                                    : `${currentTheme.cardBg} ${currentTheme.text} hover:scale-105`
                            }`}
                        >
                            {tab.replace('-', ' ')}
                        </button>
                    ))}
                </div>
            </nav>

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {activeTab === 'dashboard' && (
                    <div className="space-y-8">
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                            <div className={`${currentTheme.cardBg} rounded-xl p-6`}>
                                <h3 className={`text-lg font-semibold ${currentTheme.text}`}>Active Deployments</h3>
                                <p className={`text-3xl font-bold ${currentTheme.accent} bg-clip-text text-transparent`}>{deployments.filter(d => d.status === 'Active').length}</p>
                            </div>
                            <div className={`${currentTheme.cardBg} rounded-xl p-6`}>
                                <h3 className={`text-lg font-semibold ${currentTheme.text}`}>Completed Jobs</h3>
                                <p className={`text-3xl font-bold ${currentTheme.accent} bg-clip-text text-transparent`}>{trainingJobs.filter(j => j.status === 'Completed').length}</p>
                            </div>
                            <div className={`${currentTheme.cardBg} rounded-xl p-6`}>
                                <h3 className={`text-lg font-semibold ${currentTheme.text}`}>Models in Registry</h3>
                                <p className={`text-3xl font-bold ${currentTheme.accent} bg-clip-text text-transparent`}>{modelRegistry.length}</p>
                            </div>
                            <div className={`${currentTheme.cardBg} rounded-xl p-6`}>
                                <h3 className={`text-lg font-semibold ${currentTheme.text}`}>Datasets</h3>
                                <p className={`text-3xl font-bold ${currentTheme.accent} bg-clip-text text-transparent`}>{salesHistory ? 1 : 0}</p>
                            </div>
                        </div>
                        <div className={`${currentTheme.cardBg} rounded-xl p-6`}>
                            <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Recent Activity</h3>
                            <ul className="space-y-2">
                                {trainingJobs.slice(0, 3).map(job => (
                                    <li key={job.id} className="text-sm">
                                        <span className={currentTheme.textSecondary}>{job.startTime.toLocaleTimeString()}: </span>
                                        <span className={currentTheme.text}>Training job {job.id.substring(4, 8)} finished with status: {job.status}</span>
                                    </li>
                                ))}
                                {deployments.slice(0, 2).map(dep => (
                                    <li key={dep.id} className="text-sm">
                                        <span className={currentTheme.textSecondary}>{dep.deployedAt.toLocaleTimeString()}: </span>
                                        <span className={currentTheme.text}>Model {dep.modelVersion} was deployed.</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                )}

                {activeTab === 'data-management' && (
                    <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                        <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Data Management</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                            <div>
                                <h4 className={`text-lg font-semibold ${currentTheme.text} mb-2`}>Product Catalog</h4>
                                <button
                                    onClick={() => {
                                        const newProducts = generateSampleData();
                                        setProducts(newProducts);
                                        notifications.addNotification('New product catalog generated.', 'success');
                                    }}
                                    className={`px-4 py-2 mb-4 rounded-lg text-sm ${currentTheme.accent} text-white`}
                                >
                                    Generate New Catalog
                                </button>
                                <p className={`${currentTheme.textSecondary}`}>{products.length} products loaded.</p>
                            </div>
                            <div>
                                <h4 className={`text-lg font-semibold ${currentTheme.text} mb-2`}>Sales History Dataset</h4>
                                <button
                                    onClick={handleLoadHistory}
                                    disabled={!!salesHistory}
                                    className={`px-4 py-2 mb-4 rounded-lg text-sm ${!!salesHistory ? 'bg-gray-500' : `${currentTheme.accent} text-white`}`}
                                >
                                    {salesHistory ? 'History Loaded' : 'Generate Sales History'}
                                </button>
                                <p className={`${currentTheme.textSecondary}`}>{salesHistory ? '52 weeks of sales data loaded.' : 'No sales history loaded.'}</p>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'training-jobs' && (
                    <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                        <div className="flex justify-between items-center mb-4">
                            <h3 className={`text-xl font-bold ${currentTheme.text}`}>Training Jobs</h3>
                            <button
                                onClick={handleStartTraining}
                                disabled={!salesHistory || training.isTraining}
                                className={`px-6 py-2 rounded-lg font-medium transition-all ${!salesHistory || training.isTraining ? 'bg-gray-500 cursor-not-allowed' : `${currentTheme.accent} text-white`}`}
                            >
                                Start New Training Job
                            </button>
                        </div>
                        {training.isTraining && (
                            <div className="mb-4">
                                <div className="flex justify-between text-sm">
                                    <span className={currentTheme.textSecondary}>Training in progress... Epoch {training.currentEpoch}/{trainingConfig.epochs}</span>
                                    <span className={currentTheme.textSecondary}>{training.progress.toFixed(1)}%</span>
                                </div>
                                <div className="w-full bg-gray-700 rounded-full h-2 mt-1">
                                    <div
                                        className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                                        style={{ width: `${training.progress}%` }}
                                    ></div>
                                </div>
                            </div>
                        )}
                        <table className="min-w-full text-sm">
                            <thead>
                            <tr className="border-b border-gray-600">
                                <th className={`py-2 text-left ${currentTheme.textSecondary}`}>Job ID</th>
                                <th className={`py-2 text-left ${currentTheme.textSecondary}`}>Status</th>
                                <th className={`py-2 text-left ${currentTheme.textSecondary}`}>Duration (s)</th>
                                <th className={`py-2 text-left ${currentTheme.textSecondary}`}>Final Loss (MSE)</th>
                            </tr>
                            </thead>
                            <tbody>
                            {trainingJobs.map(job => (
                                <tr key={job.id} className="border-b border-gray-700">
                                    <td className={`py-2 ${currentTheme.text}`}>{job.id.substring(4, 10)}</td>
                                    <td className={`py-2 ${job.status === 'Completed' ? 'text-green-400' : 'text-yellow-400'}`}>{job.status}</td>
                                    <td className={`py-2 ${currentTheme.text}`}>{job.duration.toFixed(2)}</td>
                                    <td className={`py-2 ${currentTheme.text}`}>{job.finalLoss ? job.finalLoss.toFixed(2) : 'N/A'}</td>
                                </tr>
                            ))}
                            </tbody>
                        </table>
                    </div>
                )}

                {activeTab === 'model-registry' && (
                    <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                        <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Model Registry</h3>
                        <table className="min-w-full text-sm">
                            <thead>
                            <tr className="border-b border-gray-600">
                                <th className={`py-2 text-left ${currentTheme.textSecondary}`}>Model Version</th>
                                <th className={`py-2 text-left ${currentTheme.textSecondary}`}>Created At</th>
                                <th className={`py-2 text-left ${currentTheme.textSecondary}`}>Training Job</th>
                                <th className={`py-2 text-left ${currentTheme.textSecondary}`}>Loss (MSE)</th>
                                <th className={`py-2 text-left ${currentTheme.textSecondary}`}>Actions</th>
                            </tr>
                            </thead>
                            <tbody>
                            {modelRegistry.map(model => (
                                <tr key={model.version} className="border-b border-gray-700">
                                    <td className={`py-2 font-semibold ${currentTheme.text}`}>{model.version}</td>
                                    <td className={`py-2 ${currentTheme.text}`}>{model.createdAt.toLocaleString()}</td>
                                    <td className={`py-2 ${currentTheme.text}`}>{model.jobId.substring(4, 10)}</td>
                                    <td className={`py-2 ${currentTheme.text}`}>{model.loss.toFixed(2)}</td>
                                    <td className={`py-2 ${currentTheme.text}`}>
                                        <button onClick={() => handleDeployModel(model.version)} className={`px-3 py-1 rounded text-xs ${currentTheme.accent} text-white`}>Deploy</button>
                                    </td>
                                </tr>
                            ))}
                            </tbody>
                        </table>
                    </div>
                )}

                {activeTab === 'deployments' && (
                    <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                        <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Deployments</h3>
                        <table className="min-w-full text-sm">
                            <thead>
                            <tr className="border-b border-gray-600">
                                <th className={`py-2 text-left ${currentTheme.textSecondary}`}>Deployment ID</th>
                                <th className={`py-2 text-left ${currentTheme.textSecondary}`}>Model Version</th>
                                <th className={`py-2 text-left ${currentTheme.textSecondary}`}>Status</th>
                                <th className={`py-2 text-left ${currentTheme.textSecondary}`}>Deployed At</th>
                            </tr>
                            </thead>
                            <tbody>
                            {deployments.map(dep => (
                                <tr key={dep.id} className="border-b border-gray-700">
                                    <td className={`py-2 ${currentTheme.text}`}>{dep.id.substring(7,13)}</td>
                                    <td className={`py-2 font-semibold ${currentTheme.text}`}>{dep.modelVersion}</td>
                                    <td className={`py-2 ${dep.status === 'Active' ? 'text-green-400' : 'text-gray-500'}`}>{dep.status}</td>
                                    <td className={`py-2 ${currentTheme.text}`}>{dep.deployedAt.toLocaleString()}</td>
                                </tr>
                            ))}
                            </tbody>
                        </table>
                    </div>
                )}

                {activeTab === 'optimization' && (
                    <div className="space-y-8">
                        <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                            <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Run Optimization</h3>
                            <div className="mb-4">
                                <label className={`block text-sm font-medium ${currentTheme.textSecondary}`}>Select Model</label>
                                <select
                                    value={selectedModelForAnalysis || ''}
                                    onChange={(e) => setSelectedModelForAnalysis(e.target.value)}
                                    className={`mt-1 block w-full md:w-1/3 rounded-md ${currentTheme.cardBg} ${currentTheme.text} px-3 py-2`}
                                >
                                    <option value="" disabled>Select a model version</option>
                                    {modelRegistry.map(m => <option key={m.version} value={m.version}>{m.version} (Loss: {m.loss.toFixed(2)})</option>)}
                                </select>
                            </div>
                            <button
                                onClick={handleRunOptimization}
                                disabled={!selectedModelForAnalysis || optimization.isOptimizing}
                                className={`px-6 py-2 rounded-lg font-medium transition-all ${!selectedModelForAnalysis || optimization.isOptimizing ? 'bg-gray-500 cursor-not-allowed' : `${currentTheme.accent} text-white`}`}
                            >
                                Run Price Optimization
                            </button>
                        </div>
                        {optimization.currentSolution && (
                            <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                                <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Optimization Results for {selectedModelForAnalysis}</h3>
                                <div className="overflow-x-auto h-96">
                                    <table className="min-w-full">
                                        <thead className="sticky top-0 bg-slate-900/50 backdrop-blur-sm">
                                        <tr className="border-b border-gray-600">
                                            <th className={`text-left py-3 px-4 ${currentTheme.text}`}>Product</th>
                                            <th className={`text-left py-3 px-4 ${currentTheme.text}`}>Original Price</th>
                                            <th className={`text-left py-3 px-4 ${currentTheme.text}`}>Optimized Price</th>
                                            <th className={`text-left py-3 px-4 ${currentTheme.text}`}>Price Change (%)</th>
                                            <th className={`text-left py-3 px-4 ${currentTheme.text}`}>New Demand</th>
                                            <th className={`text-left py-3 px-4 ${currentTheme.text}`}>Demand Change (%)</th>
                                            <th className={`text-left py-3 px-4 ${currentTheme.text}`}>New Profit</th>
                                        </tr>
                                        </thead>
                                        <tbody>
                                        {products.slice(0, 50).map((product, index) => {
                                            const optimizedPrice = optimization.currentSolution.prices[index];
                                            const priceChangePercent = ((optimizedPrice - product.basePrice) / product.basePrice) * 100;

                                            const model = modelRegistry.find(m => m.version === selectedModelForAnalysis);
                                            const elasticityMatrix = model ? model.elasticityMatrix : {};

                                            const originalDemand = MathUtils.calculateDemand({ ...product, currentPrice: product.basePrice }, products.map(p => ({ ...p, currentPrice: p.basePrice })), elasticityMatrix);
                                            const newDemand = MathUtils.calculateDemand({ ...product, currentPrice: optimizedPrice }, products.map((p, i) => ({ ...p, currentPrice: optimization.currentSolution.prices[i] })), elasticityMatrix);

                                            const demandChangePercent = originalDemand > 0 ? ((newDemand - originalDemand) / originalDemand) * 100 : 0;

                                            const newProfit = (optimizedPrice - product.cost) * newDemand;

                                            return (
                                                <tr key={product.id} className="border-b border-gray-700 hover:bg-white/5">
                                                    <td className={`py-3 px-4 ${currentTheme.text} font-medium`}>{product.name}</td>
                                                    <td className={`py-3 px-4 ${currentTheme.text}`}>${product.basePrice.toFixed(2)}</td>
                                                    <td className={`py-3 px-4 ${currentTheme.text} font-semibold`}>${optimizedPrice.toFixed(2)}</td>
                                                    <td className={`py-3 px-4 font-semibold ${priceChangePercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                        {priceChangePercent >= 0 ? '+' : ''}{priceChangePercent.toFixed(1)}%
                                                    </td>
                                                    <td className={`py-3 px-4 ${currentTheme.text} font-semibold`}>{Math.round(newDemand)}</td>
                                                    <td className={`py-3 px-4 font-semibold ${demandChangePercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                        {demandChangePercent >= 0 ? '+' : ''}{demandChangePercent.toFixed(1)}%
                                                    </td>
                                                    <td className={`py-3 px-4 ${currentTheme.text} font-semibold`}>${newProfit.toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
                                                </tr>
                                            );
                                        })}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {activeTab === 'monte-carlo' && (
                    <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                        <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Monte Carlo Analysis</h3>
                        <div className="mb-4">
                            <label className={`block text-sm font-medium ${currentTheme.textSecondary}`}>Select Model</label>
                            <select
                                value={selectedModelForAnalysis || ''}
                                onChange={(e) => setSelectedModelForAnalysis(e.target.value)}
                                className={`mt-1 block w-full md:w-1/3 rounded-md ${currentTheme.cardBg} ${currentTheme.text} px-3 py-2`}
                            >
                                <option value="" disabled>Select a model version</option>
                                {modelRegistry.map(m => <option key={m.version} value={m.version}>{m.version} (Loss: {m.loss.toFixed(2)})</option>)}
                            </select>
                        </div>
                        <button
                            onClick={handleRunMonteCarlo}
                            disabled={!selectedModelForAnalysis || monteCarlo.isRunning}
                            className={`px-6 py-2 rounded-lg font-medium transition-all ${!selectedModelForAnalysis || monteCarlo.isRunning ? 'bg-gray-500 cursor-not-allowed' : `${currentTheme.accent} text-white`}`}
                        >
                            {monteCarlo.isRunning ? `Running... ${monteCarlo.progress.toFixed(0)}%` : 'Run Monte Carlo'}
                        </button>
                        {monteCarlo.results && (
                            <div className="mt-4">
                                <h4 className="text-lg font-semibold">Analysis Results for {selectedModelForAnalysis}</h4>
                                <div className="h-80">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <ScatterChart data={monteCarlo.results.outcomes.slice(0, 200)}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                            <XAxis dataKey="revenue" type="number" name="Revenue" stroke={currentTheme.text} />
                                            <YAxis dataKey="profit" type="number" name="Profit" stroke={currentTheme.text} />
                                            <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: 'rgba(31, 41, 55, 0.8)', border: 'none' }} />
                                            <Scatter name="Simulations" dataKey="profit" fill="#8884d8" />
                                        </ScatterChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {activeTab === 'elasticity-matrix' && (
                    <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                        <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Elasticity Matrix Visualization</h3>
                        <div className="mb-4">
                            <label className={`block text-sm font-medium ${currentTheme.textSecondary}`}>Select Model</label>
                            <select
                                value={selectedModelForAnalysis || ''}
                                onChange={(e) => setSelectedModelForAnalysis(e.target.value)}
                                className={`mt-1 block w-full md:w-1/3 rounded-md ${currentTheme.cardBg} ${currentTheme.text} px-3 py-2`}
                            >
                                <option value="" disabled>Select a model version</option>
                                {modelRegistry.map(m => <option key={m.version} value={m.version}>{m.version} (Loss: {m.loss.toFixed(2)})</option>)}
                            </select>
                        </div>
                        {modelRegistry.find(m => m.version === selectedModelForAnalysis) ? (
                            <div className="overflow-x-auto">
                                <table className="min-w-full text-xs">
                                    <thead>
                                    <tr className="border-b border-gray-600">
                                        <th className={`py-2 px-2 ${currentTheme.text}`}>Product</th>
                                        {products.slice(0, 20).map(p => <th key={p.id} className={`py-2 px-2 ${currentTheme.text}`}>{p.name}</th>)}
                                    </tr>
                                    </thead>
                                    <tbody>
                                    {products.slice(0, 20).map(p1 => (
                                        <tr key={p1.id} className="border-b border-gray-700 hover:bg-white/5">
                                            <td className={`py-2 px-2 font-medium ${currentTheme.text}`}>{p1.name}</td>
                                            {products.slice(0, 20).map(p2 => {
                                                const elasticity = modelRegistry.find(m => m.version === selectedModelForAnalysis).elasticityMatrix[p1.id]?.[p2.id] || 0;
                                                const color = p1.id === p2.id ? 'text-yellow-400' : elasticity > 0 ? 'text-green-400' : 'text-red-400';
                                                return <td key={p2.id} className={`py-2 px-2 text-center ${color}`}>{elasticity.toFixed(2)}</td>
                                            })}
                                        </tr>
                                    ))}
                                    </tbody>
                                </table>
                            </div>
                        ) : <p className={`${currentTheme.textSecondary}`}>Select a model to view its elasticity matrix.</p>}
                    </div>
                )}

                {activeTab === 'monitoring' && (
                    <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                        <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Model Monitoring</h3>
                        <p className={`${currentTheme.textSecondary}`}>Monitoring dashboard for the active deployment.</p>
                    </div>
                )}

                {activeTab === 'model-assumptions' && (
                    <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                        <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Model Assumptions</h3>
                        <div className={`space-y-4 ${currentTheme.textSecondary}`}>
                            <div>
                                <h4 className={`font-semibold ${currentTheme.text}`}>1. Demand Function</h4>
                                <p>The core demand for a product is calculated using a multiplicative model based on constant elasticity of demand. The formula is: `NewDemand = BaseDemand * (PriceRatio_1 ^ Elasticity_1) * (PriceRatio_2 ^ Elasticity_2) * ...` This assumes that the percentage change in demand due to a percentage change in price is constant.</p>
                            </div>
                            <div>
                                <h4 className={`font-semibold ${currentTheme.text}`}>2. GNN Simulation</h4>
                                <p>The EvolveGCN model is a simulation. It uses a 2-layer graph convolutional network to generate product embeddings. These embeddings capture relationships based on product features (price, demand, category) and their position in the product graph. The "learning" process is simulated by nudging a randomly initialized elasticity matrix towards a pre-calculated "ground truth" matrix to minimize Mean Squared Error against historical data.</p>
                            </div>
                            <div>
                                <h4 className={`font-semibold ${currentTheme.text}`}>3. Sales History Generation</h4>
                                <p>The historical sales data is synthetically generated. It assumes that for 52 weeks, the price of every product follows a random walk (within ±15% of its base price). The resulting demand for each week is calculated using the "ground truth" elasticity matrix, with a small amount of random noise added. This simulation does not account for seasonality, promotions, or stock-out events.</p>
                            </div>
                            <div>
                                <h4 className={`font-semibold ${currentTheme.text}`}>4. Optimization Algorithm</h4>
                                <p>The price optimization uses the Adam optimizer, a sophisticated gradient-based algorithm. It seeks to find the prices that maximize either total profit or total revenue. It assumes that the profit/revenue landscape is differentiable and has a single global maximum that can be found by following the gradient.</p>
                            </div>
                        </div>
                    </div>
                )}

            </main>

            <div className="fixed top-4 right-4 space-y-2 z-50">
                {notifications.notifications.map((notification) => (
                    <div
                        key={notification.id}
                        className={`px-4 py-3 rounded-lg shadow-lg backdrop-blur-md border transform transition-all duration-300 ${
                            notification.type === 'success'
                                ? 'bg-green-500/20 border-green-500/50 text-green-100'
                                : notification.type === 'error'
                                    ? 'bg-red-500/20 border-red-500/50 text-red-100'
                                    : 'bg-blue-500/20 border-blue-500/50 text-blue-100'
                        }`}
                    >
                        <div className="flex items-center space-x-2">
                            <span className="text-sm font-medium">{notification.message}</span>
                            <span className="text-xs opacity-75">
                {notification.timestamp.toLocaleTimeString()}
              </span>
                        </div>
                    </div>
                ))}
            </div>

            <footer className={`${currentTheme.cardBg} mt-16`}>
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                    <div className={`border-t border-gray-700 mt-8 pt-4 text-center ${currentTheme.textSecondary} text-sm`}>
                        © 2025 MLOps Platform.
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default MLOpsPlatform;
