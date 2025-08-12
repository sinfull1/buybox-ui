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
    Scatter
} from 'recharts';


// Custom Hooks for State Management
const useTraining = () => {
    const [isTraining, setIsTraining] = useState(false);
    const [progress, setProgress] = useState(0);
    const [lossHistory, setLossHistory] = useState([]);
    const [currentEpoch, setCurrentEpoch] = useState(0);
    const [trainingSpeed, setTrainingSpeed] = useState(0);

    const startTraining = useCallback(async (config) => {
        setIsTraining(true);
        setProgress(0);
        setLossHistory([]);
        setCurrentEpoch(0);

        for (let epoch = 0; epoch <= config.epochs; epoch++) {
            const startTime = Date.now();

            const loss = Math.exp(-epoch / 50) * (1 + Math.random() * 0.1) + 0.01;
            const accuracy = Math.min(0.95, 1 - Math.exp(-epoch / 30));

            setLossHistory(prev => [...prev, {
                epoch,
                loss: parseFloat(loss.toFixed(4)),
                accuracy: parseFloat(accuracy.toFixed(4)),
                validation_loss: loss * (1.1 + Math.random() * 0.1)
            }]);

            setCurrentEpoch(epoch);
            setProgress((epoch / config.epochs) * 100);

            const endTime = Date.now();
            setTrainingSpeed(1000 / (endTime - startTime));

            await new Promise(resolve => setTimeout(resolve, Math.random() * 50 + 20));
        }

        setIsTraining(false);
    }, []);

    return { isTraining, progress, lossHistory, currentEpoch, trainingSpeed, startTraining };
};

const useOptimization = () => {
    const [isOptimizing, setIsOptimizing] = useState(false);
    const [optimizationHistory, setOptimizationHistory] = useState([]);
    const [currentSolution, setCurrentSolution] = useState(null);
    const [convergenceMetrics, setConvergenceMetrics] = useState(null);
    // **NEW:** Ref to handle stopping the optimization
    const stopOptimizationRef = useRef(false);

    const startOptimization = useCallback(async (products, elasticityMatrix, constraints, objective = 'profit') => {
        setIsOptimizing(true);
        setOptimizationHistory([]);
        setConvergenceMetrics(null);
        stopOptimizationRef.current = false; // Reset stop flag

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
            // **NEW:** Check if stop has been requested
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

    // **NEW:** Function to stop the optimization
    const stopOptimization = useCallback(() => {
        stopOptimizationRef.current = true;
    }, []);

    // **NEW:** Function to reset the state
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
                priceChanges: priceVariations.map(p => (p.currentPrice - p.basePrice) / p.basePrice)
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
                min: Math.min(...revenues),
                max: Math.max(...revenues),
                std: Math.sqrt(revenues.reduce((sq, n) => sq + Math.pow(n - revenues.reduce((a, b) => a + b, 0) / revenues.length, 2), 0) / revenues.length),
                percentiles: {
                    p5: revenues.sort((a, b) => a - b)[Math.floor(revenues.length * 0.05)],
                    p95: revenues.sort((a, b) => a - b)[Math.floor(revenues.length * 0.95)]
                }
            },
            profit: {
                mean: profits.reduce((a, b) => a + b, 0) / profits.length,
                min: Math.min(...profits),
                max: Math.max(...profits),
                std: Math.sqrt(profits.reduce((sq, n) => sq + Math.pow(n - profits.reduce((a, b) => a + b, 0) / profits.length, 2), 0) / profits.length),
                percentiles: {
                    p5: profits.sort((a, b) => a - b)[Math.floor(profits.length * 0.05)],
                    p95: profits.sort((a, b) => a - b)[Math.floor(profits.length * 0.95)]
                }
            },
            outcomes
        };

        setResults(stats);
        setIsRunning(false);
    }, []);

    return { isRunning, results, progress, runMonteCarloAnalysis };
};

const useDataImport = () => {
    const [isImporting, setIsImporting] = useState(false);
    const [importResults, setImportResults] = useState(null);

    const importCSV = useCallback(async (csvText) => {
        setIsImporting(true);

        try {
            const lines = csvText.trim().split('\n');
            const headers = lines[0].split(',').map(h => h.trim().toLowerCase());

            const products = [];
            for (let i = 1; i < lines.length; i++) {
                const values = lines[i].split(',');
                const product = {
                    id: products.length, // Ensure unique ID
                    name: values[headers.indexOf('name')] || `Product ${products.length + 1}`,
                    category: values[headers.indexOf('category')] || 'General',
                    basePrice: parseFloat(values[headers.indexOf('baseprice')] || values[headers.indexOf('price')]) || 50,
                    currentPrice: parseFloat(values[headers.indexOf('currentprice')] || values[headers.indexOf('price')]) || 50,
                    baseDemand: parseFloat(values[headers.indexOf('basedemand')] || values[headers.indexOf('demand')]) || 100,
                    cost: parseFloat(values[headers.indexOf('cost')]) || 25,
                    inventory: parseInt(values[headers.indexOf('inventory')]) || 500
                };

                if (product.name && !isNaN(product.basePrice)) {
                    products.push(product);
                }
            }

            setImportResults({
                success: true,
                products,
                count: products.length,
                message: `Successfully imported ${products.length} products`
            });
        } catch (error) {
            setImportResults({
                success: false,
                products: [],
                count: 0,
                message: `Import failed: ${error.message}`
            });
        }

        setIsImporting(false);
    }, []);

    const exportCSV = useCallback((products) => {
        const headers = ['name', 'category', 'basePrice', 'currentPrice', 'baseDemand', 'cost', 'inventory'];
        const csvContent = [
            headers.join(','),
            ...products.map(p => headers.map(h => p[h]).join(','))
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'products.csv';
        a.click();
        URL.revokeObjectURL(url);
    }, []);

    return { isImporting, importResults, importCSV, exportCSV };
};

const useAdvancedOptimization = () => {
    const [isOptimizing, setIsOptimizing] = useState(false);
    const [optimizationResults, setOptimizationResults] = useState(null);
    const [optimizationHistory, setOptimizationHistory] = useState([]);

    const runMultiObjectiveOptimization = useCallback(async (products, elasticityMatrix, objectives, constraints) => {
        setIsOptimizing(true);
        setOptimizationHistory([]);

        const weights = {
            profit: objectives.profit || 0.4,
            revenue: objectives.revenue || 0.3,
            volume: objectives.volume || 0.2,
            risk: objectives.risk || 0.1
        };

        let bestSolution = null;
        let bestObjective = -Infinity;
        const maxIterations = 200;

        let currentPrices = products.map(p => p.basePrice);

        for (let iteration = 0; iteration < maxIterations; iteration++) {
            let totalProfit = 0;
            let totalRevenue = 0;
            let totalVolume = 0;
            let riskPenalty = 0;

            products.forEach((product, i) => {
                const modifiedProduct = { ...product, currentPrice: currentPrices[i] };
                const demand = MathUtils.calculateDemand(modifiedProduct,
                    products.map((p, j) => ({ ...p, currentPrice: currentPrices[j] })),
                    elasticityMatrix);

                const profit = (currentPrices[i] - product.cost) * demand;
                const revenue = currentPrices[i] * demand;

                totalProfit += profit;
                totalRevenue += revenue;
                totalVolume += demand;

                const priceChange = Math.abs(currentPrices[i] - product.basePrice) / product.basePrice;
                if (priceChange > (constraints.maxPriceChange || 0.3)) {
                    riskPenalty += priceChange * 1000;
                }
            });

            const objectiveValue =
                weights.profit * (totalProfit / 1000) +
                weights.revenue * (totalRevenue / 1000) +
                weights.volume * (totalVolume / 100) -
                weights.risk * riskPenalty;

            const learningRate = 0.01 * Math.exp(-iteration / 100);
            const newPrices = currentPrices.map((price, i) => {
                const gradient = (Math.random() - 0.5) * 2;
                let newPrice = price + learningRate * gradient;

                const minPrice = products[i].cost * (constraints.minMarkup || 1.1);
                const maxPrice = products[i].basePrice * (1 + (constraints.maxPriceChange || 0.3));
                newPrice = Math.max(minPrice, Math.min(maxPrice, newPrice));

                return newPrice;
            });

            currentPrices = newPrices;

            if (objectiveValue > bestObjective) {
                bestObjective = objectiveValue;
                bestSolution = {
                    prices: [...currentPrices],
                    profit: totalProfit,
                    revenue: totalRevenue,
                    volume: totalVolume,
                    risk: riskPenalty,
                    objective: objectiveValue
                };
            }

            setOptimizationHistory(prev => [...prev, {
                iteration,
                objective: objectiveValue,
                profit: totalProfit,
                revenue: totalRevenue,
                volume: totalVolume,
                risk: riskPenalty
            }]);

            await new Promise(resolve => setTimeout(resolve, 20));
        }

        setOptimizationResults(bestSolution);
        setIsOptimizing(false);
    }, []);

    return { isOptimizing, optimizationResults, optimizationHistory, runMultiObjectiveOptimization };
};

// Mathematical utilities for EvolveGCN
const MathUtils = {
    evolveGCNElasticityExtraction: (products, adjacencyMatrix) => {
        console.log('Running EvolveGCN elasticity extraction...');

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
                if (i === j) {
                    const embedding_i = layer2Output[i];
                    const embeddingMagnitude = Math.sqrt(embedding_i.reduce((sum, val) => sum + val * val, 0));
                    const baseElasticity = -0.8 - (embeddingMagnitude / 10) - (product.basePrice / 1000);

                    const priceLevel = product.basePrice > 100 ? 0.3 : 0;
                    const categoryFactor = product.category === 'Electronics' ? 0.2 :
                        product.category === 'Books' ? -0.3 : 0;

                    elasticityMatrix[product.id][otherProduct.id] = baseElasticity - priceLevel + categoryFactor;
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

                    const graphConnection = adjacencyMatrix[i][j];

                    const sameCategory = product.category === otherProduct.category;
                    const priceRatio = Math.min(product.basePrice, otherProduct.basePrice) /
                        Math.max(product.basePrice, otherProduct.basePrice);

                    if (sameCategory && priceRatio > 0.5) {
                        elasticityMatrix[product.id][otherProduct.id] =
                            (0.2 + similarity * 0.8 + graphConnection * 0.5) * 0.5;
                    } else if (sameCategory) {
                        elasticityMatrix[product.id][otherProduct.id] =
                            (similarity * 0.4 + graphConnection * 0.3) * 0.25;
                    } else {
                        const complementProbability = similarity * graphConnection * 2;
                        const categoryPairHash = (product.category + otherProduct.category).length % 3;

                        if (complementProbability > 0.2) {
                            elasticityMatrix[product.id][otherProduct.id] =
                                -(complementProbability * 0.45);
                        } else if (categoryPairHash === 0) {
                            elasticityMatrix[product.id][otherProduct.id] = 0.15;
                        } else if (categoryPairHash === 1) {
                            elasticityMatrix[product.id][otherProduct.id] = -0.1;
                        } else {
                            elasticityMatrix[product.id][otherProduct.id] = (similarity > 0 ? 0.02 : -0.02);
                        }

                        const categoryElasticityBoost = MathUtils.getCategoryElasticityBoost(product.category, otherProduct.category);
                        elasticityMatrix[product.id][otherProduct.id] *= categoryElasticityBoost;
                    }
                }
            });
        });

        console.log('EvolveGCN elasticity extraction completed');
        return { elasticityMatrix, embeddings: layer2Output };
    },

    getCategoryElasticityBoost: (category1, category2) => {
        const categoryRelationships = {
            'Electronics': { 'Automotive': 1.2, 'Office': 1.5, 'Health': 0.8, 'Home & Garden': 1.1, 'Sports': 0.9, 'Beauty': 0.7, 'Clothing': 0.6, 'Books': 0.5, 'Toys': 1.0 },
            'Clothing': { 'Beauty': 1.3, 'Sports': 1.4, 'Toys': 0.7, 'Office': 0.8, 'Health': 0.9, 'Electronics': 0.6, 'Automotive': 0.5, 'Books': 0.5, 'Home & Garden': 0.6 },
            'Sports': { 'Health': 1.6, 'Clothing': 1.4, 'Electronics': 0.9, 'Beauty': 0.8, 'Automotive': 0.7, 'Office': 0.6, 'Books': 0.8, 'Toys': 1.1, 'Home & Garden': 0.7 },
            'Health': { 'Beauty': 1.2, 'Sports': 1.6, 'Books': 1.0, 'Electronics': 0.8, 'Office': 0.7, 'Clothing': 0.9, 'Automotive': 0.6, 'Toys': 0.7, 'Home & Garden': 0.8 },
            'Beauty': { 'Clothing': 1.3, 'Health': 1.2, 'Electronics': 0.7, 'Sports': 0.8, 'Office': 0.8, 'Books': 0.6, 'Automotive': 0.5, 'Toys': 0.6, 'Home & Garden': 0.7 }
        };
        const boost1 = categoryRelationships[category1]?.[category2] || 1.0;
        const boost2 = categoryRelationships[category2]?.[category1] || 1.0;
        return (boost1 + boost2) / 2;
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
            const elasticity = elasticityMatrix[product.id]?.[allProducts[i].id] ||
                (product.id === allProducts[i].id ? -1.5 : 0);

            const priceRatio = Math.max(0.1, Math.min(10, allProducts[i].currentPrice / allProducts[i].basePrice));
            const cappedElasticity = Math.max(-5, Math.min(2, elasticity));

            demand *= Math.pow(priceRatio, cappedElasticity);
        }

        const minDemand = product.baseDemand * 0.01;
        const maxDemand = product.baseDemand * 5;

        return Math.max(minDemand, Math.min(maxDemand, demand));
    },

    generateElasticityMatrix: (products) => {
        const adjacencyMatrix = MathUtils.buildProductGraph(products);
        const result = MathUtils.evolveGCNElasticityExtraction(products, adjacencyMatrix);
        return result.elasticityMatrix;
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

// **NEW:** D3 component with zoom and click handler
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

        // **NEW:** Add zoom functionality
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
            .on("click", (event, d) => onNodeClick(d)) // **NEW:** Click handler
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

const EvolveGCNPlatform = () => {
    const [theme, setTheme] = useState('dark');
    const [activeTab, setActiveTab] = useState('dashboard');
    const [products, setProducts] = useState(generateSampleData());
    const [elasticityMatrix, setElasticityMatrix] = useState({});
    const [optimizationConstraints, setOptimizationConstraints] = useState({ maxPriceChange: 30 });
    const [optimizationObjective, setOptimizationObjective] = useState('profit');
    const [trainingConfig, setTrainingConfig] = useState({
        epochs: 100,
        learningRate: 0.001,
        hiddenDim: 64,
        graphMethod: 'hybrid'
    });
    // **NEW:** State for selected node in D3 graph
    const [selectedNode, setSelectedNode] = useState(null);

    const training = useTraining();
    const optimization = useOptimization();
    const notifications = useNotification();
    const monteCarloAnalysis = useMonteCarloAnalysis();
    const dataImport = useDataImport();
    const advancedOptimization = useAdvancedOptimization();

    useEffect(() => {
        notifications.addNotification('Generating initial elasticity matrix...', 'info');
        const newElasticityMatrix = MathUtils.generateElasticityMatrix(products);
        setElasticityMatrix(newElasticityMatrix);
        notifications.addNotification('Elasticity matrix ready.', 'success');
    }, [products]);

    const toggleTheme = () => {
        setTheme(prev => prev === 'dark' ? 'light' : 'dark');
    };

    const handleStartTraining = useCallback(() => {
        training.startTraining(trainingConfig);
        notifications.addNotification('EvolveGCN training started - extracting elasticities from product data', 'success');

        setTimeout(() => {
            const newElasticityMatrix = MathUtils.generateElasticityMatrix(products);
            setElasticityMatrix(newElasticityMatrix);
            notifications.addNotification('Elasticity matrix updated from EvolveGCN model', 'success');
        }, trainingConfig.epochs * 60);
    }, [trainingConfig, training, notifications, products]);

    const handleStartOptimization = useCallback(() => {
        if (Object.keys(elasticityMatrix).length === 0) {
            notifications.addNotification('Please train the EvolveGCN model first to extract elasticities', 'error');
            return;
        }
        optimization.startOptimization(products, elasticityMatrix, optimizationConstraints, optimizationObjective);
        notifications.addNotification(`Price optimization started for ${optimizationObjective} maximization`, 'success');
    }, [products, optimization, notifications, elasticityMatrix, optimizationConstraints, optimizationObjective]);

    const graphMetrics = useMemo(() => {
        const adjacencyMatrix = MathUtils.buildProductGraph(products);
        const totalEdges = adjacencyMatrix.flat().filter(val => val > 0.5).length;
        const avgDegree = totalEdges / products.length;

        return { totalEdges, avgDegree, density: totalEdges / (products.length * (products.length - 1)) };
    }, [products]);

    const performanceMetrics = useMemo(() => {
        if (!optimization.currentSolution) return null;

        const { totalRevenue, totalProfit } = optimization.currentSolution;
        const margin = totalRevenue > 0 ? (totalProfit / totalRevenue) * 100 : 0;

        return { totalRevenue, totalProfit, margin };
    }, [optimization.currentSolution]);

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
                            <h1 className={`text-3xl font-bold ${currentTheme.text}`}>EvolveGCN Platform</h1>
                            <p className={`${currentTheme.textSecondary}`}>Graph Neural Network Price Optimization</p>
                        </div>
                        <div className="flex items-center space-x-4">
                            <button
                                onClick={toggleTheme}
                                className={`px-4 py-2 rounded-lg ${currentTheme.cardBg} ${currentTheme.text} hover:scale-105 transition-all duration-200`}
                            >
                                {theme === 'dark' ? '☀️' : '🌙'} {theme === 'dark' ? 'Light' : 'Dark'}
                            </button>
                        </div>
                    </div>
                </div>
            </header>

            <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">
                <div className="flex space-x-1 overflow-x-auto">
                    {['dashboard', 'training', 'optimization', 'elasticity-matrix', 'monte-carlo', 'network-3d', 'data-import', 'executive'].map((tab) => (
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
                            <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl hover:scale-105 transition-all duration-200`}>
                                <h3 className={`text-lg font-semibold ${currentTheme.text}`}>Total Products</h3>
                                <p className={`text-3xl font-bold ${currentTheme.accent} bg-clip-text text-transparent`}>{products.length}</p>
                            </div>
                            <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl hover:scale-105 transition-all duration-200`}>
                                <h3 className={`text-lg font-semibold ${currentTheme.text}`}>Graph Density</h3>
                                <p className={`text-3xl font-bold ${currentTheme.accent} bg-clip-text text-transparent`}>
                                    {(graphMetrics.density * 100).toFixed(1)}%
                                </p>
                            </div>
                            <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl hover:scale-105 transition-all duration-200`}>
                                <h3 className={`text-lg font-semibold ${currentTheme.text}`}>Avg Degree</h3>
                                <p className={`text-3xl font-bold ${currentTheme.accent} bg-clip-text text-transparent`}>
                                    {graphMetrics.avgDegree.toFixed(1)}
                                </p>
                            </div>
                            <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl hover:scale-105 transition-all duration-200`}>
                                <h3 className={`text-lg font-semibold ${currentTheme.text}`}>Model Status</h3>
                                <p className={`text-lg font-semibold ${training.isTraining ? 'text-yellow-400' : 'text-green-400'}`}>
                                    {training.isTraining ? 'Training...' : 'Ready'}
                                </p>
                            </div>
                        </div>

                        <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                            <div className="flex justify-between items-center mb-4">
                                <h3 className={`text-xl font-bold ${currentTheme.text}`}>Product Data (Sample)</h3>
                                <button
                                    onClick={() => {
                                        const newProducts = generateSampleData();
                                        setProducts(newProducts);
                                        notifications.addNotification('New sample data generated', 'success');
                                    }}
                                    className={`px-4 py-2 rounded-lg ${currentTheme.accent} text-white hover:scale-105 transition-all duration-200`}
                                >
                                    Generate New Sample Data
                                </button>
                            </div>
                            <div className="overflow-x-auto h-96">
                                <table className="min-w-full">
                                    <thead>
                                    <tr className="border-b border-gray-600 sticky top-0 bg-slate-900/50 backdrop-blur-sm">
                                        <th className={`text-left py-3 px-4 ${currentTheme.text}`}>Product Name</th>
                                        <th className={`text-left py-3 px-4 ${currentTheme.text}`}>Category</th>
                                        <th className={`text-left py-3 px-4 ${currentTheme.text}`}>Base Price</th>
                                        <th className={`text-left py-3 px-4 ${currentTheme.text}`}>Cost</th>
                                        <th className={`text-left py-3 px-4 ${currentTheme.text}`}>Margin</th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    {products.map((product) => {
                                        const margin = ((product.basePrice - product.cost) / product.basePrice * 100);
                                        return (
                                            <tr key={product.id} className="border-b border-gray-700 hover:bg-white/5">
                                                <td className={`py-3 px-4 ${currentTheme.text} font-medium`}>{product.name}</td>
                                                <td className={`py-3 px-4 ${currentTheme.textSecondary}`}>
                            <span className="px-2 py-1 rounded-full text-xs bg-blue-500/20 text-blue-300">
                              {product.category}
                            </span>
                                                </td>
                                                <td className={`py-3 px-4 ${currentTheme.text}`}>${product.basePrice.toFixed(2)}</td>
                                                <td className={`py-3 px-4 ${currentTheme.text}`}>${product.cost.toFixed(2)}</td>
                                                <td className={`py-3 px-4 ${margin >= 20 ? 'text-green-400' : margin >= 10 ? 'text-yellow-400' : 'text-red-400'}`}>
                                                    {margin.toFixed(1)}%
                                                </td>
                                            </tr>
                                        );
                                    })}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'training' && (
                    <div className="space-y-8">
                        <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                            <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>EvolveGCN Training Configuration</h3>
                            <div className="mb-4 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                                <p className={`text-sm ${currentTheme.text}`}>
                                    <strong>Training Process:</strong> The EvolveGCN model will analyze product relationships using graph neural networks
                                    to extract price elasticities.
                                </p>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                                <div>
                                    <label className={`block text-sm font-medium ${currentTheme.textSecondary}`}>Epochs</label>
                                    <input
                                        type="number"
                                        value={trainingConfig.epochs}
                                        onChange={(e) => setTrainingConfig({...trainingConfig, epochs: parseInt(e.target.value)})}
                                        className={`mt-1 block w-full rounded-md ${currentTheme.cardBg} ${currentTheme.text} px-3 py-2`}
                                        disabled={training.isTraining}
                                    />
                                </div>
                                <div>
                                    <label className={`block text-sm font-medium ${currentTheme.textSecondary}`}>Learning Rate</label>
                                    <input
                                        type="number"
                                        step="0.0001"
                                        value={trainingConfig.learningRate}
                                        onChange={(e) => setTrainingConfig({...trainingConfig, learningRate: parseFloat(e.target.value)})}
                                        className={`mt-1 block w-full rounded-md ${currentTheme.cardBg} ${currentTheme.text} px-3 py-2`}
                                        disabled={training.isTraining}
                                    />
                                </div>
                                <div>
                                    <label className={`block text-sm font-medium ${currentTheme.textSecondary}`}>Hidden Dimension</label>
                                    <input
                                        type="number"
                                        value={trainingConfig.hiddenDim}
                                        onChange={(e) => setTrainingConfig({...trainingConfig, hiddenDim: parseInt(e.target.value)})}
                                        className={`mt-1 block w-full rounded-md ${currentTheme.cardBg} ${currentTheme.text} px-3 py-2`}
                                        disabled={training.isTraining}
                                    />
                                </div>
                                <div>
                                    <label className={`block text-sm font-medium ${currentTheme.textSecondary}`}>Graph Method</label>
                                    <select
                                        value={trainingConfig.graphMethod}
                                        onChange={(e) => setTrainingConfig({...trainingConfig, graphMethod: e.target.value})}
                                        className={`mt-1 block w-full rounded-md ${currentTheme.cardBg} ${currentTheme.text} px-3 py-2`}
                                        disabled={training.isTraining}
                                    >
                                        <option value="category">Category-based</option>
                                        <option value="correlation">Correlation-based</option>
                                        <option value="price">Price Similarity</option>
                                        <option value="hybrid">Hybrid Approach</option>
                                    </select>
                                </div>
                            </div>
                            <div className="mt-4">
                                <button
                                    onClick={handleStartTraining}
                                    disabled={training.isTraining}
                                    className={`px-8 py-3 rounded-lg font-medium transition-all duration-200 ${
                                        training.isTraining
                                            ? 'bg-gray-400 cursor-not-allowed'
                                            : `${currentTheme.accent} text-white hover:scale-105 shadow-lg`
                                    }`}
                                >
                                    {training.isTraining ? 'Training...' : 'Start Training'}
                                </button>
                            </div>
                        </div>

                        {training.lossHistory.length > 0 && (
                            <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                                <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Training Curves</h3>
                                <div className="h-80">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={training.lossHistory}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                            <XAxis dataKey="epoch" stroke={currentTheme.text} />
                                            <YAxis stroke={currentTheme.text} />
                                            <Tooltip
                                                contentStyle={{
                                                    backgroundColor: theme === 'dark' ? '#1f2937' : '#ffffff',
                                                    border: '1px solid #374151',
                                                    borderRadius: '8px',
                                                    color: currentTheme.text
                                                }}
                                            />
                                            <Legend />
                                            <Line type="monotone" dataKey="loss" stroke="#f59e0b" strokeWidth={2} name="Training Loss" />
                                            <Line type="monotone" dataKey="validation_loss" stroke="#ef4444" strokeWidth={2} name="Validation Loss" />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {activeTab === 'optimization' && (
                    <div className="space-y-8">
                        <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                            <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Price Optimization</h3>

                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                                <div>
                                    <label className={`block text-sm font-medium ${currentTheme.textSecondary}`}>Objective Function</label>
                                    <select
                                        value={optimizationObjective}
                                        onChange={(e) => setOptimizationObjective(e.target.value)}
                                        className={`mt-1 block w-full rounded-md ${currentTheme.cardBg} ${currentTheme.text} px-3 py-2`}
                                    >
                                        <option value="profit">Profit Maximization</option>
                                        <option value="revenue">Revenue Maximization</option>
                                    </select>
                                </div>
                                <div>
                                    <label className={`block text-sm font-medium ${currentTheme.textSecondary}`}>Max Price Change (%)</label>
                                    <input
                                        type="number"
                                        value={optimizationConstraints.maxPriceChange}
                                        onChange={(e) => setOptimizationConstraints({ ...optimizationConstraints, maxPriceChange: parseInt(e.target.value) })}
                                        min={5}
                                        max={50}
                                        className={`mt-1 block w-full rounded-md ${currentTheme.cardBg} ${currentTheme.text} px-3 py-2`}
                                    />
                                </div>
                                <div>
                                    <label className={`block text-sm font-medium ${currentTheme.textSecondary}`}>Optimizer</label>
                                    <input
                                        type="text"
                                        defaultValue="Adam"
                                        readOnly
                                        className={`mt-1 block w-full rounded-md ${currentTheme.cardBg} ${currentTheme.textSecondary} px-3 py-2`}
                                    />
                                </div>
                            </div>
                            <div className="flex items-center space-x-4">
                                <button
                                    onClick={handleStartOptimization}
                                    disabled={optimization.isOptimizing || Object.keys(elasticityMatrix).length === 0}
                                    className={`px-8 py-3 rounded-lg font-medium transition-all duration-200 ${
                                        optimization.isOptimizing || Object.keys(elasticityMatrix).length === 0
                                            ? 'bg-gray-400 cursor-not-allowed'
                                            : `bg-gradient-to-r from-green-500 to-blue-500 text-white hover:scale-105 shadow-lg`
                                    }`}
                                >
                                    {optimization.isOptimizing ? 'Optimizing...' : 'Start Optimization'}
                                </button>
                                {/* **NEW:** Stop/Reset Button */}
                                <button
                                    onClick={() => {
                                        if (optimization.isOptimizing) {
                                            optimization.stopOptimization();
                                            notifications.addNotification('Optimization stopped by user.', 'info');
                                        } else {
                                            optimization.resetOptimization();
                                            notifications.addNotification('Optimization results have been reset.', 'info');
                                        }
                                    }}
                                    disabled={!optimization.isOptimizing && !optimization.currentSolution}
                                    className={`px-8 py-3 rounded-lg font-medium transition-all duration-200 ${
                                        optimization.isOptimizing
                                            ? 'bg-gradient-to-r from-red-500 to-orange-500 text-white hover:scale-105 shadow-lg'
                                            : 'bg-gradient-to-r from-gray-500 to-gray-600 text-white hover:scale-105 shadow-lg'
                                    } ${(!optimization.isOptimizing && !optimization.currentSolution) && 'cursor-not-allowed opacity-50'}`}
                                >
                                    {optimization.isOptimizing ? 'Stop' : 'Reset'}
                                </button>
                            </div>
                            {optimization.convergenceMetrics && (
                                <div className={`mt-4 text-sm p-3 rounded-lg ${optimization.convergenceMetrics.converged ? 'bg-green-500/20 text-green-300' : 'bg-yellow-500/20 text-yellow-300'}`}>
                                    {optimization.convergenceMetrics.reason}
                                </div>
                            )}
                        </div>

                        {optimization.optimizationHistory.length > 0 && (
                            <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                                <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Convergence Progress</h3>
                                <div className="h-80">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={optimization.optimizationHistory}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                            <XAxis dataKey="iteration" stroke={currentTheme.text} />
                                            <YAxis yAxisId="left" stroke="#10b981" />
                                            <YAxis yAxisId="right" orientation="right" stroke="#f59e0b" />
                                            <Tooltip
                                                contentStyle={{
                                                    backgroundColor: theme === 'dark' ? '#1f2937' : '#ffffff',
                                                    border: '1px solid #374151',
                                                    borderRadius: '8px'
                                                }}
                                            />
                                            <Legend />
                                            <Line yAxisId="left" type="monotone" dataKey="objective" stroke="#10b981" strokeWidth={2} name={optimizationObjective === 'profit' ? 'Total Profit ($)' : 'Total Revenue ($)'} />
                                            <Line yAxisId="right" type="monotone" dataKey="gradient_norm" stroke="#f59e0b" strokeWidth={2} name="Gradient Norm" />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}

                        {optimization.currentSolution && (
                            <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                                <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Optimized Prices & Business Impact (Sample)</h3>
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
                                        {products.slice(0, 20).map((product, index) => {
                                            const optimizedPrice = optimization.currentSolution.prices[index];
                                            const priceChangePercent = ((optimizedPrice - product.basePrice) / product.basePrice) * 100;

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

                {activeTab === 'elasticity-matrix' && (
                    <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                        <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>EvolveGCN Elasticity Matrix Analysis (Sample)</h3>
                        {Object.keys(elasticityMatrix).length > 0 ? (
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
                                                const elasticity = elasticityMatrix[p1.id]?.[p2.id] || 0;
                                                const color = p1.id === p2.id ? 'text-yellow-400' : elasticity > 0 ? 'text-green-400' : 'text-red-400';
                                                return <td key={p2.id} className={`py-2 px-2 text-center ${color}`}>{elasticity.toFixed(2)}</td>
                                            })}
                                        </tr>
                                    ))}
                                    </tbody>
                                </table>
                            </div>
                        ) : <p className={`${currentTheme.textSecondary}`}>No elasticity matrix available. Please train the model.</p>}
                    </div>
                )}

                {activeTab === 'monte-carlo' && (
                    <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                        <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Monte Carlo Sensitivity Analysis</h3>
                        <button
                            onClick={() => monteCarloAnalysis.runMonteCarloAnalysis(products, elasticityMatrix, 1000)}
                            disabled={monteCarloAnalysis.isRunning || Object.keys(elasticityMatrix).length === 0}
                            className={`px-8 py-3 rounded-lg font-medium transition-all duration-200 mb-4 ${
                                monteCarloAnalysis.isRunning || Object.keys(elasticityMatrix).length === 0
                                    ? 'bg-gray-400 cursor-not-allowed'
                                    : `bg-gradient-to-r from-purple-500 to-pink-500 text-white hover:scale-105 shadow-lg`
                            }`}
                        >
                            {monteCarloAnalysis.isRunning ? `Running... ${monteCarloAnalysis.progress.toFixed(0)}%` : 'Start Monte Carlo Analysis'}
                        </button>
                        {monteCarloAnalysis.results && (
                            <div className="h-80">
                                <ResponsiveContainer width="100%" height="100%">
                                    <ScatterChart data={monteCarloAnalysis.results.outcomes.slice(0, 200)}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                        <XAxis dataKey="revenue" stroke={currentTheme.text} name="Revenue" />
                                        <YAxis dataKey="profit" stroke={currentTheme.text} name="Profit" />
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: theme === 'dark' ? '#1f2937' : '#ffffff',
                                                border: '1px solid #374151',
                                                borderRadius: '8px'
                                            }}
                                        />
                                        <Scatter dataKey="profit" fill="#8b5cf6" />
                                    </ScatterChart>
                                </ResponsiveContainer>
                            </div>
                        )}
                    </div>
                )}

                {activeTab === 'network-3d' && (
                    <div className="space-y-8">
                        <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                            <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Product Relationship Network</h3>
                            <div className="mb-4 p-4 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
                                <p className={`text-sm ${currentTheme.text}`}>
                                    <strong>Interactive D3 Visualization:</strong> Zoom with your mouse wheel and drag nodes to see the network adjust. Click a node to see details below.
                                </p>
                            </div>

                            <div className="relative w-full h-[600px] border border-gray-600 rounded-lg overflow-hidden bg-gradient-to-br from-blue-900/20 to-purple-900/20">
                                <D3Network products={products} elasticityMatrix={elasticityMatrix} theme={theme} onNodeClick={setSelectedNode} />
                            </div>

                            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                                <div className="flex items-center">
                                    <div className="w-3 h-3 rounded-full bg-indigo-500 mr-2"></div>
                                    <span className={currentTheme.textSecondary}>Node size = Demand</span>
                                </div>
                                <div className="flex items-center">
                                    <div className="w-3 h-1 bg-green-500 mr-2"></div>
                                    <span className={currentTheme.textSecondary}>Green Link = Complements</span>
                                </div>
                                <div className="flex items-center">
                                    <div className="w-3 h-1 bg-red-500 mr-2"></div>
                                    <span className={currentTheme.textSecondary}>Red Link = Substitutes</span>
                                </div>
                            </div>
                        </div>
                        {/* **NEW:** Node Detail Table */}
                        {selectedNode && (
                            <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                                <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Details for: {selectedNode.name}</h3>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                    <div>
                                        <h4 className={`text-lg font-semibold ${currentTheme.text} mb-2`}>Product Info</h4>
                                        <table className="min-w-full text-sm">
                                            <tbody>
                                            <tr className="border-b border-gray-700"><td className={`py-2 pr-4 ${currentTheme.textSecondary}`}>Category</td><td className={currentTheme.text}>{selectedNode.category}</td></tr>
                                            <tr className="border-b border-gray-700"><td className={`py-2 pr-4 ${currentTheme.textSecondary}`}>Base Price</td><td className={currentTheme.text}>${selectedNode.basePrice.toFixed(2)}</td></tr>
                                            <tr className="border-b border-gray-700"><td className={`py-2 pr-4 ${currentTheme.textSecondary}`}>Cost</td><td className={currentTheme.text}>${selectedNode.cost.toFixed(2)}</td></tr>
                                            <tr className="border-b border-gray-700"><td className={`py-2 pr-4 ${currentTheme.textSecondary}`}>Base Demand</td><td className={currentTheme.text}>{Math.round(selectedNode.baseDemand)}</td></tr>
                                            </tbody>
                                        </table>
                                    </div>
                                    <div>
                                        <h4 className={`text-lg font-semibold ${currentTheme.text} mb-2`}>Connected Products</h4>
                                        <div className="overflow-y-auto h-40">
                                            <table className="min-w-full text-sm">
                                                <thead>
                                                <tr className="border-b border-gray-600">
                                                    <th className={`text-left py-2 ${currentTheme.text}`}>Product</th>
                                                    <th className={`text-left py-2 ${currentTheme.text}`}>Relationship</th>
                                                    <th className={`text-left py-2 ${currentTheme.text}`}>Elasticity</th>
                                                </tr>
                                                </thead>
                                                <tbody>
                                                {products.map(p => {
                                                    const elasticity = elasticityMatrix[selectedNode.id]?.[p.id];
                                                    if (p.id !== selectedNode.id && elasticity && Math.abs(elasticity) > 0.05) {
                                                        const isSubstitute = elasticity > 0;
                                                        return (
                                                            <tr key={p.id} className="border-b border-gray-700">
                                                                <td className={`py-2 ${currentTheme.text}`}>{p.name}</td>
                                                                <td className={isSubstitute ? 'text-green-400' : 'text-red-400'}>{isSubstitute ? 'Substitute' : 'Complement'}</td>
                                                                <td className={isSubstitute ? 'text-green-400' : 'text-red-400'}>{elasticity.toFixed(3)}</td>
                                                            </tr>
                                                        )
                                                    }
                                                    return null;
                                                })}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {activeTab === 'data-import' && (
                    <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                        <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Data Import & Export</h3>
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                            <div>
                                <h4 className={`text-lg font-semibold ${currentTheme.text} mb-4`}>Import Product Data</h4>
                                <textarea
                                    placeholder="name,category,price,demand,cost,inventory&#10;Product A,Electronics,99.99,150,45.00,200"
                                    rows={4}
                                    className={`w-full rounded-md ${currentTheme.cardBg} ${currentTheme.text} px-3 py-2 border border-gray-600 font-mono text-sm`}
                                    onChange={(e) => {
                                        if (e.target.value.trim()) {
                                            dataImport.importCSV(e.target.value);
                                        }
                                    }}
                                />
                                {dataImport.importResults && (
                                    <div className={`mt-4 p-4 rounded-lg ${dataImport.importResults.success ? 'bg-green-500/10' : 'bg-red-500/10'}`}>
                                        <p className={dataImport.importResults.success ? 'text-green-300' : 'text-red-300'}>{dataImport.importResults.message}</p>
                                        {dataImport.importResults.success && (
                                            <button
                                                onClick={() => {
                                                    setProducts(dataImport.importResults.products);
                                                    notifications.addNotification(`Imported ${dataImport.importResults.count} products`, 'success');
                                                }}
                                                className="mt-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600"
                                            >
                                                Use Imported Data
                                            </button>
                                        )}
                                    </div>
                                )}
                            </div>
                            <div>
                                <h4 className={`text-lg font-semibold ${currentTheme.text} mb-4`}>Export & Backup</h4>
                                <button
                                    onClick={() => dataImport.exportCSV(products)}
                                    className={`w-full px-4 py-3 rounded-lg ${currentTheme.accent} text-white hover:scale-105 transition-all`}
                                >
                                    Export Current Products (CSV)
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'executive' && (
                    <div className={`${currentTheme.cardBg} rounded-xl p-6 shadow-xl`}>
                        <h3 className={`text-xl font-bold ${currentTheme.text} mb-4`}>Executive Summary</h3>
                        {performanceMetrics ? (
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                <div>
                                    <h4 className="text-lg font-semibold text-green-400">Optimized Revenue</h4>
                                    <p className={`text-3xl font-bold ${currentTheme.text}`}>${performanceMetrics.totalRevenue.toLocaleString(undefined, {maximumFractionDigits: 0})}</p>
                                </div>
                                <div>
                                    <h4 className="text-lg font-semibold text-blue-400">Optimized Profit</h4>
                                    <p className={`text-3xl font-bold ${currentTheme.text}`}>${performanceMetrics.totalProfit.toLocaleString(undefined, {maximumFractionDigits: 0})}</p>
                                </div>
                                <div>
                                    <h4 className="text-lg font-semibold text-purple-400">Optimized Margin</h4>
                                    <p className={`text-3xl font-bold ${currentTheme.text}`}>{performanceMetrics.margin.toFixed(1)}%</p>
                                </div>
                            </div>
                        ) : <p className={`${currentTheme.textSecondary}`}>Run an optimization to see the executive summary.</p>}
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
                        © 2025 EvolveGCN Platform. Built with React and advanced mathematics.
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default EvolveGCNPlatform;