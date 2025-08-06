// components/PricingSensitivityVisualization.jsx
"use client"

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { ChevronDown, ChevronUp, DollarSign, TrendingUp, Package, BarChart3, Network, Settings, Play, RefreshCw, Save, X } from 'lucide-react';
import * as d3 from 'd3';

// Product categories with colors
const CATEGORIES = {
    'Electronics': '#3b82f6',
    'Accessories': '#10b981',
    'Software': '#f59e0b',
    'Services': '#ef4444',
    'Peripherals': '#8b5cf6'
};

// Generate sample products
const generateProducts = (count = 20) => {
    const categories = Object.keys(CATEGORIES);
    const products = [];

    const productNames = [
        'Laptop Pro', 'Wireless Mouse', 'USB-C Hub', 'Webcam HD', 'Keyboard Mech',
        'Monitor 4K', 'Headphones BT', 'Laptop Stand', 'External SSD', 'Graphics Tablet',
        'Smartphone', 'Tablet Pro', 'Smart Watch', 'Earbuds Pro', 'Charging Station',
        'Office Suite', 'VPN Service', 'Cloud Storage', 'Laptop Bag', 'Screen Protector'
    ];

    for (let i = 0; i < count; i++) {
        const category = categories[Math.floor(Math.random() * categories.length)];
        const basePrice = Math.random() * 500 + 50;
        const elasticity = -(0.8 + Math.random() * 1.2);

        // Generate elasticity curve data points
        const elasticityCurve = [];
        for (let priceRatio = 0.5; priceRatio <= 2.0; priceRatio += 0.1) {
            const demandRatio = Math.pow(priceRatio, elasticity);
            elasticityCurve.push({
                priceRatio: priceRatio,
                demandRatio: demandRatio,
                price: basePrice * priceRatio,
                demand: 100 * demandRatio
            });
        }

        products.push({
            id: `product_${i}`,
            name: productNames[i] || `Product ${i}`,
            category: category,
            basePrice: Math.round(basePrice),
            currentPrice: Math.round(basePrice),
            cost: Math.round(basePrice * (0.4 + Math.random() * 0.3)),
            elasticity: elasticity,
            elasticityCurve: elasticityCurve,
            inventory: Math.floor(Math.random() * 200 + 20),
            demand: 100
        });
    }

    return products;
};

// Generate relationships between products with cross-elasticity curves
const generateRelationships = (products) => {
    const relationships = [];

    products.forEach((source, i) => {
        products.forEach((target, j) => {
            if (i !== j) {
                let strength = 0;
                let type = 'independent';
                let crossElasticity = 0;

                if (source.category === target.category) {
                    // Same category - substitutes
                    strength = 0.3 + Math.random() * 0.4;
                    type = 'substitute';
                    crossElasticity = strength;
                } else if (Math.random() < 0.3) {
                    // Random complementary relationships
                    strength = 0.1 + Math.random() * 0.3;
                    type = 'complement';
                    crossElasticity = -strength * 0.5;
                } else {
                    strength = Math.random() * 0.1;
                    crossElasticity = strength * 0.1;
                }

                // Generate cross-elasticity curve
                const crossElasticityCurve = [];
                for (let priceRatio = 0.5; priceRatio <= 2.0; priceRatio += 0.1) {
                    const effectRatio = 1 + crossElasticity * (priceRatio - 1);
                    crossElasticityCurve.push({
                        priceRatio: priceRatio,
                        effectRatio: effectRatio
                    });
                }

                relationships.push({
                    source: source.id,
                    target: target.id,
                    strength: strength,
                    type: type,
                    crossElasticity: crossElasticity,
                    crossElasticityCurve: crossElasticityCurve
                });
            }
        });
    });

    return relationships;
};

// Spline-based Elasticity Curve Editor Component
const ElasticityCurveEditor = ({ product, onSave, onClose }) => {
    const svgRef = useRef(null);
    const [controlPoints, setControlPoints] = useState([]);
    const [isDragging, setIsDragging] = useState(false);

    // Initialize control points from elasticity curve
    useEffect(() => {
        // Create fewer control points for better spline editing
        const points = [
            { x: 0.5, y: Math.pow(0.5, product.elasticity), id: 0 },
            { x: 0.7, y: Math.pow(0.7, product.elasticity), id: 1 },
            { x: 1.0, y: 1.0, id: 2 }, // Base point (100% price, 100% demand)
            { x: 1.3, y: Math.pow(1.3, product.elasticity), id: 3 },
            { x: 1.6, y: Math.pow(1.6, product.elasticity), id: 4 },
            { x: 2.0, y: Math.pow(2.0, product.elasticity), id: 5 }
        ];
        setControlPoints(points);
    }, [product.elasticity]);

    useEffect(() => {
        if (!controlPoints.length) return;

        const svg = d3.select(svgRef.current);
        svg.selectAll("*").remove();

        const margin = { top: 20, right: 30, bottom: 50, left: 60 };
        const width = 500 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;

        const g = svg
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Scales
        const xScale = d3.scaleLinear()
            .domain([0.5, 2.0])
            .range([0, width]);

        const yScale = d3.scaleLinear()
            .domain([0, 2.5])
            .range([height, 0]);

        // Grid
        const xGrid = g.append("g")
            .attr("class", "grid")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale)
                .tickValues([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
                .tickSize(-height)
                .tickFormat(""))
            .style("stroke", "#e0e0e0")
            .style("stroke-dasharray", "2,2");

        const yGrid = g.append("g")
            .attr("class", "grid")
            .call(d3.axisLeft(yScale)
                .tickValues([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])
                .tickSize(-width)
                .tickFormat(""))
            .style("stroke", "#e0e0e0")
            .style("stroke-dasharray", "2,2");

        // Axes
        g.append("g")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale)
                .tickValues([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
                .tickFormat(d => `${(d * 100).toFixed(0)}%`))
            .selectAll("text")
            .style("font-size", "10px");

        g.append("g")
            .call(d3.axisLeft(yScale)
                .tickValues([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])
                .tickFormat(d => `${(d * 100).toFixed(0)}%`))
            .selectAll("text")
            .style("font-size", "10px");

        // Axis labels
        g.append("text")
            .attr("x", width / 2)
            .attr("y", height + 35)
            .attr("text-anchor", "middle")
            .style("font-size", "12px")
            .style("font-weight", "bold")
            .text("Price Ratio");

        g.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", -45)
            .attr("x", -height / 2)
            .attr("text-anchor", "middle")
            .style("font-size", "12px")
            .style("font-weight", "bold")
            .text("Demand Ratio");

        // Reference lines at 100%
        g.append("line")
            .attr("x1", xScale(1))
            .attr("x2", xScale(1))
            .attr("y1", 0)
            .attr("y2", height)
            .attr("stroke", "#999")
            .attr("stroke-width", 2)
            .attr("stroke-dasharray", "5,5");

        g.append("line")
            .attr("x1", 0)
            .attr("x2", width)
            .attr("y1", yScale(1))
            .attr("y2", yScale(1))
            .attr("stroke", "#999")
            .attr("stroke-width", 2)
            .attr("stroke-dasharray", "5,5");

        // Create spline generator
        const spline = d3.line()
            .x(d => xScale(d.x))
            .y(d => yScale(d.y))
            .curve(d3.curveCatmullRom.alpha(0.5)); // Smooth catmull-rom spline

        // Draw the spline curve
        const curvePath = g.append("path")
            .datum(controlPoints)
            .attr("fill", "none")
            .attr("stroke", "#3b82f6")
            .attr("stroke-width", 3)
            .attr("d", spline);

        // Add control points
        const points = g.selectAll(".control-point")
            .data(controlPoints)
            .enter().append("circle")
            .attr("class", "control-point")
            .attr("cx", d => xScale(d.x))
            .attr("cy", d => yScale(d.y))
            .attr("r", 7)
            .attr("fill", "#3b82f6")
            .attr("stroke", "#fff")
            .attr("stroke-width", 3)
            .attr("cursor", "ns-resize")
            .style("user-select", "none");

        // Add control point labels
        const labels = g.selectAll(".point-label")
            .data(controlPoints)
            .enter().append("text")
            .attr("class", "point-label")
            .attr("x", d => xScale(d.x))
            .attr("y", d => yScale(d.y) - 15)
            .attr("text-anchor", "middle")
            .attr("font-size", "11px")
            .attr("font-weight", "bold")
            .attr("fill", "#666")
            .text(d => `${(d.y * 100).toFixed(0)}%`);

        // Add drag behavior
        points.call(d3.drag()
            .on("start", function(event, d) {
                setIsDragging(true);
                d3.select(this)
                    .attr("r", 9)
                    .attr("fill", "#1d4ed8")
                    .style("cursor", "ns-resize");
            })
            .on("drag", function(event, d) {
                // Only allow vertical movement
                const newY = event.y;
                const constrainedY = Math.max(0, Math.min(height, newY));

                // Convert to data value
                let newDemandRatio = yScale.invert(constrainedY);
                newDemandRatio = Math.max(0.1, Math.min(2.5, newDemandRatio));

                // Update control point
                d.y = newDemandRatio;

                // Update visual position
                d3.select(this).attr("cy", yScale(newDemandRatio));

                // Update spline curve
                curvePath.attr("d", spline);

                // Update labels
                labels.data(controlPoints)
                    .attr("x", d => xScale(d.x))
                    .attr("y", d => yScale(d.y) - 15)
                    .text(d => `${(d.y * 100).toFixed(0)}%`);

                // Update state
                setControlPoints([...controlPoints]);
            })
            .on("end", function(event, d) {
                setIsDragging(false);
                d3.select(this)
                    .attr("r", 7)
                    .attr("fill", "#3b82f6")
                    .style("cursor", "ns-resize");
            }));

        // Add elasticity display
        const currentElasticity = calculateElasticityFromPoints(controlPoints);
        g.append("text")
            .attr("x", width - 10)
            .attr("y", 15)
            .attr("text-anchor", "end")
            .attr("font-size", "14px")
            .attr("font-weight", "bold")
            .attr("fill", "#333")
            .text(`Elasticity: ${currentElasticity.toFixed(2)}`);

        // Add curve type indicator
        g.append("text")
            .attr("x", 10)
            .attr("y", 15)
            .attr("font-size", "12px")
            .attr("font-weight", "bold")
            .attr("fill", "#666")
            .text("Catmull-Rom Spline");

    }, [controlPoints, product.elasticity]);

    // Calculate elasticity from control points
    const calculateElasticityFromPoints = (points) => {
        // Find points around 100% price ratio
        const basePoint = points.find(p => Math.abs(p.x - 1.0) < 0.1);
        const nextPoint = points.find(p => p.x > 1.0 && p.x < 1.4);

        if (basePoint && nextPoint) {
            const priceChange = (nextPoint.x - basePoint.x) / basePoint.x;
            const demandChange = (nextPoint.y - basePoint.y) / basePoint.y;
            return demandChange / priceChange;
        }
        return product.elasticity; // fallback
    };

    // Generate curve points from spline for saving
    const generateCurveFromSpline = () => {
        const spline = d3.line()
            .x(d => d.x)
            .y(d => d.y)
            .curve(d3.curveCatmullRom.alpha(0.5));

        // Generate dense points along the spline
        const curvePoints = [];
        for (let ratio = 0.5; ratio <= 2.0; ratio += 0.05) {
            // Interpolate y value at this x position using the control points
            let y = interpolateSpline(controlPoints, ratio);
            curvePoints.push({
                priceRatio: ratio,
                demandRatio: y,
                price: product.basePrice * ratio,
                demand: 100 * y
            });
        }
        return curvePoints;
    };

    // Spline interpolation helper
    const interpolateSpline = (points, x) => {
        // Simple linear interpolation between nearest points
        // In a production app, you'd use a proper spline interpolation
        const sorted = points.sort((a, b) => a.x - b.x);

        if (x <= sorted[0].x) return sorted[0].y;
        if (x >= sorted[sorted.length - 1].x) return sorted[sorted.length - 1].y;

        for (let i = 0; i < sorted.length - 1; i++) {
            if (x >= sorted[i].x && x <= sorted[i + 1].x) {
                const t = (x - sorted[i].x) / (sorted[i + 1].x - sorted[i].x);
                return sorted[i].y + t * (sorted[i + 1].y - sorted[i].y);
            }
        }
        return 1.0; // fallback
    };

    const handleSave = () => {
        const newElasticity = calculateElasticityFromPoints(controlPoints);
        const newCurve = generateCurveFromSpline();

        onSave({
            ...product,
            elasticity: newElasticity,
            elasticityCurve: newCurve
        });
    };

    return (
        <div className="space-y-4">
            <div className="bg-gray-50 p-4 rounded-lg border">
                <svg ref={svgRef}></svg>
            </div>
            <div className="flex justify-between items-center">
                <div className="text-sm text-gray-600">
                    <div className="font-medium mb-1">Spline Curve Editor</div>
                    <div>• Drag control points vertically to adjust demand response</div>
                    <div>• Smooth Catmull-Rom spline interpolation</div>
                    <div>• {controlPoints.length} control points for precise editing</div>
                </div>
                <div className="flex gap-2">
                    <Button variant="outline" onClick={onClose}>Cancel</Button>
                    <Button onClick={handleSave}>
                        <Save className="h-4 w-4 mr-2" />
                        Save Curve
                    </Button>
                </div>
            </div>
        </div>
    );
};

// Cross-Elasticity Curve Editor
const CrossElasticityCurveEditor = ({ sourceProduct, targetProduct, relationship, onSave, onClose }) => {
    const svgRef = useRef(null);
    const [curvePoints, setCurvePoints] = useState([...relationship.crossElasticityCurve]);

    useEffect(() => {
        if (!curvePoints.length) return;

        const svg = d3.select(svgRef.current);
        svg.selectAll("*").remove();

        const margin = { top: 20, right: 30, bottom: 40, left: 50 };
        const width = 500 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;

        const g = svg
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Scales
        const xScale = d3.scaleLinear()
            .domain([0.5, 2])
            .range([0, width]);

        const yScale = d3.scaleLinear()
            .domain([0.5, 1.5])
            .range([height, 0]);

        // Axes
        g.append("g")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale).tickFormat(d => `${(d * 100).toFixed(0)}%`))
            .append("text")
            .attr("x", width / 2)
            .attr("y", 35)
            .attr("fill", "black")
            .style("text-anchor", "middle")
            .text(`${sourceProduct.name} Price Ratio`);

        g.append("g")
            .call(d3.axisLeft(yScale).tickFormat(d => `${(d * 100).toFixed(0)}%`))
            .append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", -35)
            .attr("x", -height / 2)
            .attr("fill", "black")
            .style("text-anchor", "middle")
            .text(`${targetProduct.name} Demand Effect`);

        // Reference lines
        g.append("line")
            .attr("x1", xScale(1))
            .attr("x2", xScale(1))
            .attr("y1", 0)
            .attr("y2", height)
            .attr("stroke", "#ccc")
            .attr("stroke-dasharray", "3,3");

        g.append("line")
            .attr("x1", 0)
            .attr("x2", width)
            .attr("y1", yScale(1))
            .attr("y2", yScale(1))
            .attr("stroke", "#ccc")
            .attr("stroke-dasharray", "3,3");

        // Line generator
        const line = d3.line()
            .x(d => xScale(d.priceRatio))
            .y(d => yScale(d.effectRatio))
            .curve(d3.curveMonotoneX);

        // Draw curve
        const path = g.append("path")
            .datum(curvePoints)
            .attr("fill", "none")
            .attr("stroke", relationship.type === 'substitute' ? "#ef4444" : "#10b981")
            .attr("stroke-width", 2)
            .attr("d", line);

        // Add draggable points for cross-elasticity
        g.selectAll(".point")
            .data(curvePoints)
            .enter().append("circle")
            .attr("class", "point")
            .attr("cx", d => xScale(d.priceRatio))
            .attr("cy", d => yScale(d.effectRatio))
            .attr("r", 6)
            .attr("fill", relationship.type === 'substitute' ? "#ef4444" : "#10b981")
            .attr("stroke", "#fff")
            .attr("stroke-width", 2)
            .attr("cursor", "ns-resize")
            .style("user-select", "none")
            .call(d3.drag()
                .on("start", function(event, d) {
                    d3.select(this)
                        .attr("r", 8)
                        .style("cursor", "ns-resize");
                })
                .on("drag", function(event, d) {
                    // Use event.y directly which is already in the correct coordinate system
                    const newY = event.y;

                    // Constrain to chart area
                    const constrainedY = Math.max(0, Math.min(height, newY));

                    // Convert to data value
                    let newEffectRatio = yScale.invert(constrainedY);

                    // Allow free movement with reasonable bounds
                    newEffectRatio = Math.max(0.3, Math.min(1.7, newEffectRatio));

                    // Update data
                    d.effectRatio = newEffectRatio;

                    // Update visual position using the DATA VALUE, not screen coordinate
                    d3.select(this).attr("cy", yScale(newEffectRatio));

                    // Redraw path
                    path.attr("d", line);

                    // Update state
                    setCurvePoints([...curvePoints]);
                })
                .on("end", function(event, d) {
                    d3.select(this)
                        .attr("r", 6)
                        .style("cursor", "ns-resize");
                }));

        // Add type and strength display
        g.append("text")
            .attr("x", width - 10)
            .attr("y", 10)
            .attr("text-anchor", "end")
            .attr("font-size", "14px")
            .text(`Type: ${relationship.type} | Cross-elasticity: ${relationship.crossElasticity.toFixed(3)}`);

    }, [curvePoints, sourceProduct, targetProduct, relationship]);

    const handleSave = () => {
        // Calculate new cross-elasticity from curve
        const midPoint = curvePoints.find(p => Math.abs(p.priceRatio - 1) < 0.05);
        const nextPoint = curvePoints.find(p => Math.abs(p.priceRatio - 1.1) < 0.05);

        if (midPoint && nextPoint) {
            const priceChange = 0.1; // 10% price change
            const effectChange = nextPoint.effectRatio - midPoint.effectRatio;
            const newCrossElasticity = effectChange / priceChange;

            onSave({
                ...relationship,
                crossElasticity: newCrossElasticity,
                crossElasticityCurve: curvePoints,
                type: newCrossElasticity > 0 ? 'substitute' : 'complement'
            });
        }
    };

    return (
        <div className="space-y-4">
            <div className="bg-gray-50 p-4 rounded">
                <svg ref={svgRef}></svg>
            </div>
            <div className="flex justify-between items-center">
                <div className="text-sm text-gray-600">
                    Shows how {sourceProduct.name} price changes affect {targetProduct.name} demand
                </div>
                <div className="flex gap-2">
                    <Button variant="outline" onClick={onClose}>Cancel</Button>
                    <Button onClick={handleSave}>
                        <Save className="h-4 w-4 mr-2" />
                        Save Curve
                    </Button>
                </div>
            </div>
        </div>
    );
};

// Enhanced Sensitivity Matrix with Hover and Click
const SensitivityMatrix = ({ products, relationships, onCellClick }) => {
    const canvasRef = useRef(null);
    const [tooltip, setTooltip] = useState({ show: false, x: 0, y: 0, value: '', products: null });
    const [hoveredCell, setHoveredCell] = useState(null);

    useEffect(() => {
        if (!products.length) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const size = products.length;
        const cellSize = 25;
        const margin = 100;

        canvas.width = size * cellSize + margin + 50;
        canvas.height = size * cellSize + margin + 50;

        // Clear canvas
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Create sensitivity matrix
        const matrix = Array(size).fill().map(() => Array(size).fill(0));

        products.forEach((p, i) => {
            matrix[i][i] = p.elasticity;
        });

        relationships.forEach(rel => {
            const sourceIdx = products.findIndex(p => p.id === rel.source);
            const targetIdx = products.findIndex(p => p.id === rel.target);
            if (sourceIdx !== -1 && targetIdx !== -1) {
                matrix[targetIdx][sourceIdx] = rel.crossElasticity;
            }
        });

        // Color scale
        const colorScale = d3.scaleSequential()
            .domain([-1, 1])
            .interpolator(d3.interpolateRdBu);

        // Draw cells
        matrix.forEach((row, i) => {
            row.forEach((value, j) => {
                const x = j * cellSize + margin;
                const y = i * cellSize + margin;

                ctx.fillStyle = colorScale(value);
                ctx.fillRect(x, y, cellSize - 1, cellSize - 1);

                // Highlight hovered cell
                if (hoveredCell && hoveredCell.i === i && hoveredCell.j === j) {
                    ctx.strokeStyle = '#000';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x, y, cellSize - 1, cellSize - 1);
                }
            });
        });

        // Add labels
        ctx.fillStyle = 'black';
        ctx.font = '10px sans-serif';

        // X-axis labels (rotated)
        ctx.save();
        products.forEach((p, i) => {
            ctx.save();
            ctx.translate(margin + i * cellSize + cellSize/2, margin - 5);
            ctx.rotate(-Math.PI / 4);
            ctx.fillText(p.name.substring(0, 10), 0, 0);
            ctx.restore();
        });
        ctx.restore();

        // Y-axis labels
        products.forEach((p, i) => {
            ctx.fillText(p.name.substring(0, 10), 10, margin + i * cellSize + cellSize/2 + 3);
        });

        // Add axis titles
        ctx.font = '12px sans-serif';
        ctx.fillText('Target Product (Affected)', 5, margin - 20);

        ctx.save();
        ctx.translate(margin - 20, canvas.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Source Product (Price Changed)', 0, 0);
        ctx.restore();

    }, [products, relationships, hoveredCell]);

    const handleMouseMove = (e) => {
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const cellSize = 25;
        const margin = 100;

        const j = Math.floor((x - margin) / cellSize);
        const i = Math.floor((y - margin) / cellSize);

        if (i >= 0 && i < products.length && j >= 0 && j < products.length) {
            setHoveredCell({ i, j });

            let value;
            if (i === j) {
                value = products[i].elasticity;
            } else {
                const rel = relationships.find(r =>
                    r.source === products[j].id && r.target === products[i].id
                );
                value = rel ? rel.crossElasticity : 0;
            }

            // Position tooltip to the right of the matrix if there's space, otherwise to the left
            const canvasWidth = canvas.width;
            const tooltipX = rect.right + 10; // Position to the right of the canvas
            const tooltipY = rect.top + y; // Align vertically with mouse

            setTooltip({
                show: true,
                x: tooltipX,
                y: tooltipY,
                value: value.toFixed(3),
                products: {
                    source: products[j],
                    target: products[i],
                    isDiagonal: i === j
                }
            });
        } else {
            setHoveredCell(null);
            setTooltip({ show: false, x: 0, y: 0, value: '', products: null });
        }
    };

    const handleClick = (e) => {
        if (hoveredCell && tooltip.products) {
            onCellClick(tooltip.products.source, tooltip.products.target, tooltip.products.isDiagonal);
        }
    };

    return (
        <div className="relative">
            <canvas
                ref={canvasRef}
                onMouseMove={handleMouseMove}
                onMouseLeave={() => {
                    setHoveredCell(null);
                    setTooltip({ show: false, x: 0, y: 0, value: '', products: null });
                }}
                onClick={handleClick}
                className="cursor-pointer"
            />
            {tooltip.show && (
                <div
                    className="fixed z-10 bg-gray-900 text-white p-2 rounded text-sm pointer-events-none shadow-lg"
                    style={{ left: tooltip.x, top: tooltip.y }}
                >
                    {tooltip.products?.isDiagonal ? (
                        <div>
                            <div className="font-bold">{tooltip.products.source.name}</div>
                            <div>Own Elasticity: {tooltip.value}</div>
                            <div className="text-xs mt-1 text-gray-300">Click to edit curve</div>
                        </div>
                    ) : (
                        <div>
                            <div className="font-bold text-xs">{tooltip.products?.source.name} → {tooltip.products?.target.name}</div>
                            <div>Cross-Elasticity: {tooltip.value}</div>
                            <div className="text-xs mt-1 text-gray-300">Click to edit relationship</div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

// Network Graph Component
const NetworkGraph = ({ products, relationships, selectedProduct, onSelectProduct, priceChanges }) => {
    const svgRef = useRef(null);

    useEffect(() => {
        if (!products.length) return;

        const svg = d3.select(svgRef.current);
        svg.selectAll("*").remove();

        const width = 800;
        const height = 600;

        const g = svg
            .attr("viewBox", `0 0 ${width} ${height}`)
            .append("g");

        // Create zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.5, 3])
            .on("zoom", (event) => {
                g.attr("transform", event.transform);
            });

        svg.call(zoom);

        // Prepare nodes and links
        const nodes = products.map(p => ({
            ...p,
            priceChange: priceChanges[p.id] || 0
        }));

        const links = relationships.filter(r =>
            Math.abs(r.strength) > 0.1
        );

        // Add initial positioning to spread nodes better
        nodes.forEach((node, i) => {
            const angle = (i / nodes.length) * 2 * Math.PI;
            const radius = Math.min(width, height) * 0.3;
            node.x = width / 2 + Math.cos(angle) * radius;
            node.y = height / 2 + Math.sin(angle) * radius;
        });

        // Create force simulation with boundary constraints
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links)
                .id(d => d.id)
                .distance(d => 100 * (1 - d.strength))
                .strength(d => d.strength * 0.5))
            .force("charge", d3.forceManyBody().strength(-200))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(35))
            .force("x", d3.forceX(width / 2).strength(0.05))
            .force("y", d3.forceY(height / 2).strength(0.05));

        // Add links
        const link = g.append("g")
            .selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("stroke", d => d.type === 'substitute' ? "#ef4444" : "#10b981")
            .attr("stroke-opacity", d => d.strength)
            .attr("stroke-width", d => Math.max(1, d.strength * 5));

        // Add nodes
        const node = g.append("g")
            .selectAll("g")
            .data(nodes)
            .enter().append("g")
            .attr("cursor", "pointer")
            .on("click", (event, d) => onSelectProduct(d.id))
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        // Add circles for nodes
        node.append("circle")
            .attr("r", d => 20 + Math.abs(d.elasticity) * 10)
            .attr("fill", d => CATEGORIES[d.category])
            .attr("stroke", d => selectedProduct === d.id ? "#000" : "#fff")
            .attr("stroke-width", d => selectedProduct === d.id ? 3 : 2)
            .attr("opacity", 0.9);

        // Add price change indicator
        node.append("circle")
            .attr("r", d => Math.abs(d.priceChange) * 2)
            .attr("fill", d => d.priceChange > 0 ? "#22c55e" : "#ef4444")
            .attr("opacity", 0.5);

        // Add labels
        node.append("text")
            .text(d => d.name.split(' ')[0])
            .attr("font-size", "10px")
            .attr("text-anchor", "middle")
            .attr("dy", 35)
            .attr("fill", "#374151");

        // Add price labels
        node.append("text")
            .text(d => `${d.currentPrice}`)
            .attr("font-size", "12px")
            .attr("font-weight", "bold")
            .attr("text-anchor", "middle")
            .attr("dy", -25)
            .attr("fill", "#111827");

        // Update positions on simulation tick with boundary constraints
        simulation.on("tick", () => {
            // Keep nodes within boundaries
            nodes.forEach(d => {
                const radius = 20 + Math.abs(d.elasticity) * 10;
                d.x = Math.max(radius + 10, Math.min(width - radius - 10, d.x));
                d.y = Math.max(radius + 10, Math.min(height - radius - 10, d.y));
            });

            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node.attr("transform", d => `translate(${d.x},${d.y})`);
        });

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            const radius = 20 + Math.abs(d.elasticity) * 10;
            d.fx = Math.max(radius + 10, Math.min(width - radius - 10, event.x));
            d.fy = Math.max(radius + 10, Math.min(height - radius - 10, event.y));
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
        }

    }, [products, relationships, selectedProduct, priceChanges]);

    return <svg ref={svgRef} className="w-full h-full" />;
};

// Main Component
export default function PricingSensitivityVisualization() {
    const [products, setProducts] = useState([]);
    const [relationships, setRelationships] = useState([]);
    const [selectedProduct, setSelectedProduct] = useState(null);
    const [isSimulating, setIsSimulating] = useState(false);
    const [metrics, setMetrics] = useState({
        totalRevenue: 0,
        totalProfit: 0,
        totalDemand: 0,
        avgSensitivity: 0
    });
    const [showSettings, setShowSettings] = useState(false);
    const [priceMultiplier, setPriceMultiplier] = useState(1.0);
    const [elasticityEditorOpen, setElasticityEditorOpen] = useState(false);
    const [crossElasticityEditorOpen, setCrossElasticityEditorOpen] = useState(false);
    const [editingProduct, setEditingProduct] = useState(null);
    const [editingRelationship, setEditingRelationship] = useState(null);
    const [editingSourceProduct, setEditingSourceProduct] = useState(null);
    const [editingTargetProduct, setEditingTargetProduct] = useState(null);

    // Initialize products and relationships
    useEffect(() => {
        const prods = generateProducts(20);
        const rels = generateRelationships(prods);
        setProducts(prods);
        setRelationships(rels);
        calculateMetrics(prods);
    }, []);

    // Calculate price changes for visualization
    const priceChanges = useMemo(() => {
        const changes = {};
        products.forEach(p => {
            changes[p.id] = ((p.currentPrice - p.basePrice) / p.basePrice) * 100;
        });
        return changes;
    }, [products]);

    // Calculate metrics
    const calculateMetrics = (productList) => {
        const revenue = productList.reduce((sum, p) => sum + p.currentPrice * p.demand, 0);
        const profit = productList.reduce((sum, p) => sum + (p.currentPrice - p.cost) * p.demand, 0);
        const demand = productList.reduce((sum, p) => sum + p.demand, 0);
        const avgSens = productList.reduce((sum, p) => sum + Math.abs(p.elasticity), 0) / productList.length;

        setMetrics({
            totalRevenue: revenue,
            totalProfit: profit,
            totalDemand: demand,
            avgSensitivity: avgSens
        });
    };

    // Update single product price
    const updateProductPrice = (productId, newPrice) => {
        setProducts(prev => {
            const updated = prev.map(p => {
                if (p.id === productId) {
                    const priceChange = (newPrice - p.basePrice) / p.basePrice;
                    const demandChange = 1 + p.elasticity * priceChange;

                    // Calculate cross effects
                    let crossEffect = 1;
                    relationships
                        .filter(r => r.target === p.id)
                        .forEach(rel => {
                            const sourceProduct = prev.find(sp => sp.id === rel.source);
                            if (sourceProduct) {
                                const sourcePriceChange = (sourceProduct.currentPrice - sourceProduct.basePrice) / sourceProduct.basePrice;
                                crossEffect *= (1 + rel.crossElasticity * sourcePriceChange);
                            }
                        });

                    return {
                        ...p,
                        currentPrice: Math.round(newPrice),
                        demand: Math.max(10, Math.round(100 * demandChange * crossEffect))
                    };
                }
                return p;
            });

            calculateMetrics(updated);
            return updated;
        });
    };

    // Handle matrix cell click
    const handleMatrixCellClick = (sourceProduct, targetProduct, isDiagonal) => {
        if (isDiagonal) {
            // Edit own elasticity
            setEditingProduct(sourceProduct);
            setElasticityEditorOpen(true);
        } else {
            // Edit cross-elasticity
            const relationship = relationships.find(r =>
                r.source === sourceProduct.id && r.target === targetProduct.id
            );
            if (relationship) {
                setEditingSourceProduct(sourceProduct);
                setEditingTargetProduct(targetProduct);
                setEditingRelationship(relationship);
                setCrossElasticityEditorOpen(true);
            }
        }
    };

    // Save elasticity changes
    const handleSaveElasticity = (updatedProduct) => {
        setProducts(prev => prev.map(p =>
            p.id === updatedProduct.id ? updatedProduct : p
        ));
        setElasticityEditorOpen(false);
        setEditingProduct(null);

        // Recalculate metrics with new elasticity
        recalculateWithNewElasticities();
    };

    // Save cross-elasticity changes
    const handleSaveCrossElasticity = (updatedRelationship) => {
        setRelationships(prev => prev.map(r =>
            (r.source === updatedRelationship.source && r.target === updatedRelationship.target)
                ? updatedRelationship : r
        ));
        setCrossElasticityEditorOpen(false);
        setEditingRelationship(null);
        setEditingSourceProduct(null);
        setEditingTargetProduct(null);

        // Recalculate metrics with new relationships
        recalculateWithNewElasticities();
    };

    // Recalculate all demands with new elasticities
    const recalculateWithNewElasticities = () => {
        setProducts(prev => {
            const updated = prev.map(p => {
                const priceChange = (p.currentPrice - p.basePrice) / p.basePrice;
                const demandChange = 1 + p.elasticity * priceChange;

                // Calculate cross effects with updated relationships
                let crossEffect = 1;
                relationships
                    .filter(r => r.target === p.id)
                    .forEach(rel => {
                        const sourceProduct = prev.find(sp => sp.id === rel.source);
                        if (sourceProduct) {
                            const sourcePriceChange = (sourceProduct.currentPrice - sourceProduct.basePrice) / sourceProduct.basePrice;
                            crossEffect *= (1 + rel.crossElasticity * sourcePriceChange);
                        }
                    });

                return {
                    ...p,
                    demand: Math.max(10, Math.round(100 * demandChange * crossEffect))
                };
            });

            calculateMetrics(updated);
            return updated;
        });
    };

    // Run price optimization simulation
    const runSimulation = () => {
        setIsSimulating(true);

        // Simple gradient-based optimization
        const optimizeStep = () => {
            setProducts(prev => {
                const updated = prev.map(p => {
                    // Calculate profit gradient
                    const currentProfit = (p.currentPrice - p.cost) * p.demand;
                    const testPrice = p.currentPrice * 1.01;
                    const testDemand = p.demand * (1 + p.elasticity * 0.01);
                    const testProfit = (testPrice - p.cost) * testDemand;

                    const gradient = testProfit - currentProfit;

                    // Update price based on gradient
                    let newPrice = p.currentPrice;
                    if (gradient > 0) {
                        newPrice = Math.min(p.basePrice * 2, p.currentPrice * 1.02);
                    } else if (gradient < 0) {
                        newPrice = Math.max(p.cost * 1.1, p.currentPrice * 0.98);
                    }

                    const priceChange = (newPrice - p.basePrice) / p.basePrice;
                    const newDemand = Math.max(10, Math.round(100 * (1 + p.elasticity * priceChange)));

                    return {
                        ...p,
                        currentPrice: Math.round(newPrice),
                        demand: newDemand
                    };
                });

                calculateMetrics(updated);
                return updated;
            });
        };

        // Run optimization steps
        let steps = 0;
        const interval = setInterval(() => {
            optimizeStep();
            steps++;
            if (steps >= 10) {
                clearInterval(interval);
                setIsSimulating(false);
            }
        }, 500);
    };

    // Reset all prices
    const resetPrices = () => {
        setProducts(prev => {
            const updated = prev.map(p => ({
                ...p,
                currentPrice: p.basePrice,
                demand: 100
            }));
            calculateMetrics(updated);
            return updated;
        });
    };

    // Apply multiplier to all prices
    const applyPriceMultiplier = () => {
        setProducts(prev => {
            const updated = prev.map(p => {
                const newPrice = Math.round(p.basePrice * priceMultiplier);
                const priceChange = (newPrice - p.basePrice) / p.basePrice;
                const newDemand = Math.max(10, Math.round(100 * (1 + p.elasticity * priceChange)));

                return {
                    ...p,
                    currentPrice: newPrice,
                    demand: newDemand
                };
            });
            calculateMetrics(updated);
            return updated;
        });
    };

    const selectedProductData = selectedProduct ? products.find(p => p.id === selectedProduct) : null;

    return (
        <div className="w-full max-w-7xl mx-auto p-4 space-y-4">
            {/* Header */}
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Network className="h-6 w-6" />
              Ecommerce Pricing Sensitivity Analysis
            </span>
                        <div className="flex gap-2">
                            <Button
                                onClick={runSimulation}
                                disabled={isSimulating}
                                size="sm"
                                className="gap-2"
                            >
                                <Play className="h-4 w-4" />
                                {isSimulating ? 'Optimizing...' : 'Optimize Prices'}
                            </Button>
                            <Button
                                onClick={resetPrices}
                                variant="outline"
                                size="sm"
                                className="gap-2"
                            >
                                <RefreshCw className="h-4 w-4" />
                                Reset
                            </Button>
                            <Button
                                onClick={() => setShowSettings(!showSettings)}
                                variant="outline"
                                size="sm"
                            >
                                <Settings className="h-4 w-4" />
                            </Button>
                        </div>
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    {/* Metrics Dashboard */}
                    <div className="grid grid-cols-4 gap-4 mb-4">
                        <div className="bg-blue-50 p-3 rounded-lg">
                            <div className="flex items-center gap-2 text-blue-600 mb-1">
                                <DollarSign className="h-4 w-4" />
                                <span className="text-sm font-medium">Revenue</span>
                            </div>
                            <div className="text-2xl font-bold text-blue-900">
                                ${metrics.totalRevenue.toLocaleString()}
                            </div>
                        </div>
                        <div className="bg-green-50 p-3 rounded-lg">
                            <div className="flex items-center gap-2 text-green-600 mb-1">
                                <TrendingUp className="h-4 w-4" />
                                <span className="text-sm font-medium">Profit</span>
                            </div>
                            <div className="text-2xl font-bold text-green-900">
                                ${metrics.totalProfit.toLocaleString()}
                            </div>
                        </div>
                        <div className="bg-purple-50 p-3 rounded-lg">
                            <div className="flex items-center gap-2 text-purple-600 mb-1">
                                <Package className="h-4 w-4" />
                                <span className="text-sm font-medium">Total Demand</span>
                            </div>
                            <div className="text-2xl font-bold text-purple-900">
                                {metrics.totalDemand.toLocaleString()}
                            </div>
                        </div>
                        <div className="bg-orange-50 p-3 rounded-lg">
                            <div className="flex items-center gap-2 text-orange-600 mb-1">
                                <BarChart3 className="h-4 w-4" />
                                <span className="text-sm font-medium">Avg Sensitivity</span>
                            </div>
                            <div className="text-2xl font-bold text-orange-900">
                                {metrics.avgSensitivity.toFixed(2)}
                            </div>
                        </div>
                    </div>

                    {/* Settings Panel */}
                    {showSettings && (
                        <Alert className="mb-4">
                            <AlertDescription>
                                <div className="space-y-3">
                                    <div>
                                        <label className="text-sm font-medium">Global Price Multiplier: {priceMultiplier.toFixed(2)}x</label>
                                        <div className="flex gap-2 mt-1">
                                            <Slider
                                                value={[priceMultiplier]}
                                                onValueChange={([v]) => setPriceMultiplier(v)}
                                                min={0.5}
                                                max={2.0}
                                                step={0.05}
                                                className="flex-1"
                                            />
                                            <Button size="sm" onClick={applyPriceMultiplier}>Apply</Button>
                                        </div>
                                    </div>
                                    <div className="text-xs text-gray-600">
                                        • Node size represents price elasticity (larger = more sensitive)
                                        • Green lines show complementary products, red shows substitutes
                                        • Click on products to adjust individual prices
                                        • Click matrix cells to edit elasticity curves
                                    </div>
                                </div>
                            </AlertDescription>
                        </Alert>
                    )}
                </CardContent>
            </Card>

            {/* Main Content */}
            <div className="grid grid-cols-3 gap-4">
                {/* Network Visualization */}
                <div className="col-span-2">
                    <Card className="h-[600px]">
                        <CardHeader>
                            <CardTitle className="text-lg">Product Network</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <NetworkGraph
                                products={products}
                                relationships={relationships}
                                selectedProduct={selectedProduct}
                                onSelectProduct={setSelectedProduct}
                                priceChanges={priceChanges}
                            />
                        </CardContent>
                    </Card>
                </div>

                {/* Product Details Panel */}
                <div className="col-span-1">
                    <Card className="h-[600px]">
                        <CardHeader>
                            <CardTitle className="text-lg">
                                {selectedProductData ? selectedProductData.name : 'Select a Product'}
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            {selectedProductData ? (
                                <div className="space-y-4">
                                    <div>
                                        <span className="text-sm text-gray-500">Category</span>
                                        <div className="font-medium">{selectedProductData.category}</div>
                                    </div>

                                    <div>
                                        <span className="text-sm text-gray-500">Current Price</span>
                                        <div className="text-2xl font-bold">${selectedProductData.currentPrice}</div>
                                        <div className="text-sm text-gray-500">
                                            Base: ${selectedProductData.basePrice} | Cost: ${selectedProductData.cost}
                                        </div>
                                    </div>

                                    <div>
                                        <label className="text-sm font-medium">Adjust Price</label>
                                        <Slider
                                            value={[selectedProductData.currentPrice]}
                                            onValueChange={([v]) => updateProductPrice(selectedProductData.id, v)}
                                            min={selectedProductData.cost}
                                            max={selectedProductData.basePrice * 2}
                                            step={1}
                                            className="mt-2"
                                        />
                                    </div>

                                    <div className="grid grid-cols-2 gap-3">
                                        <div>
                                            <span className="text-sm text-gray-500">Elasticity</span>
                                            <div className="font-medium">{selectedProductData.elasticity.toFixed(2)}</div>
                                        </div>
                                        <div>
                                            <span className="text-sm text-gray-500">Current Demand</span>
                                            <div className="font-medium">{selectedProductData.demand}</div>
                                        </div>
                                        <div>
                                            <span className="text-sm text-gray-500">Inventory</span>
                                            <div className="font-medium">{selectedProductData.inventory}</div>
                                        </div>
                                        <div>
                                            <span className="text-sm text-gray-500">Margin</span>
                                            <div className="font-medium">
                                                {((selectedProductData.currentPrice - selectedProductData.cost) / selectedProductData.currentPrice * 100).toFixed(1)}%
                                            </div>
                                        </div>
                                    </div>

                                    <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={() => {
                                            setEditingProduct(selectedProductData);
                                            setElasticityEditorOpen(true);
                                        }}
                                    >
                                        Edit Elasticity Curve
                                    </Button>

                                    <div>
                                        <span className="text-sm font-medium">Related Products</span>
                                        <div className="mt-2 space-y-1">
                                            {relationships
                                                .filter(r => r.source === selectedProductData.id && r.strength > 0.2)
                                                .sort((a, b) => b.strength - a.strength)
                                                .slice(0, 5)
                                                .map(rel => {
                                                    const relProduct = products.find(p => p.id === rel.target);
                                                    return relProduct ? (
                                                        <div key={rel.target} className="flex justify-between text-sm">
                                                            <span>{relProduct.name}</span>
                                                            <span className={rel.type === 'substitute' ? 'text-red-500' : 'text-green-500'}>
                                {rel.type} ({(rel.strength * 100).toFixed(0)}%)
                              </span>
                                                        </div>
                                                    ) : null;
                                                })}
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="text-gray-500 text-center mt-8">
                                    Click on a product node to view and adjust its details
                                </div>
                            )}
                        </CardContent>
                    </Card>
                </div>
            </div>

            {/* Sensitivity Analysis Tabs */}
            <Card>
                <CardContent className="pt-6">
                    <Tabs defaultValue="matrix">
                        <TabsList>
                            <TabsTrigger value="matrix">Sensitivity Matrix</TabsTrigger>
                            <TabsTrigger value="rankings">Sensitivity Rankings</TabsTrigger>
                            <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
                        </TabsList>

                        <TabsContent value="matrix">
                            <div className="overflow-auto">
                                <SensitivityMatrix
                                    products={products}
                                    relationships={relationships}
                                    onCellClick={handleMatrixCellClick}
                                />
                            </div>
                        </TabsContent>

                        <TabsContent value="rankings">
                            <div className="space-y-2">
                                <h3 className="font-medium mb-2">Products by Price Sensitivity</h3>
                                {products
                                    .sort((a, b) => Math.abs(b.elasticity) - Math.abs(a.elasticity))
                                    .slice(0, 10)
                                    .map(p => (
                                        <div key={p.id} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                                            <span className="font-medium">{p.name}</span>
                                            <div className="flex gap-4 text-sm">
                                                <span>Elasticity: {p.elasticity.toFixed(2)}</span>
                                                <span className="text-gray-500">${p.currentPrice}</span>
                                            </div>
                                        </div>
                                    ))}
                            </div>
                        </TabsContent>

                        <TabsContent value="recommendations">
                            <div className="space-y-3">
                                <Alert>
                                    <AlertDescription>
                                        <strong>Optimization Opportunities:</strong>
                                        <ul className="mt-2 space-y-1 text-sm">
                                            <li>• Products with high margins and low elasticity can support price increases</li>
                                            <li>• Bundle complementary products for increased revenue</li>
                                            <li>• Avoid simultaneous discounts on substitute products</li>
                                            <li>• Focus promotions on products with high positive cross-elasticities</li>
                                        </ul>
                                    </AlertDescription>
                                </Alert>

                                <div>
                                    <h3 className="font-medium mb-2">Specific Recommendations:</h3>
                                    {products
                                        .filter(p => (p.currentPrice - p.cost) / p.currentPrice > 0.5 && Math.abs(p.elasticity) < 1)
                                        .slice(0, 3)
                                        .map(p => (
                                            <div key={p.id} className="p-2 bg-green-50 rounded mb-2">
                        <span className="text-sm">
                          <strong>{p.name}</strong>: Consider 5-10% price increase (low elasticity, high margin)
                        </span>
                                            </div>
                                        ))}
                                </div>
                            </div>
                        </TabsContent>
                    </Tabs>
                </CardContent>
            </Card>

            {/* Elasticity Editor Dialog */}
            <Dialog open={elasticityEditorOpen} onOpenChange={setElasticityEditorOpen}>
                <DialogContent className="max-w-3xl">
                    <DialogHeader>
                        <DialogTitle>Edit Price Elasticity Curve: {editingProduct?.name}</DialogTitle>
                    </DialogHeader>
                    {editingProduct && (
                        <ElasticityCurveEditor
                            product={editingProduct}
                            onSave={handleSaveElasticity}
                            onClose={() => setElasticityEditorOpen(false)}
                        />
                    )}
                </DialogContent>
            </Dialog>

            {/* Cross-Elasticity Editor Dialog */}
            <Dialog open={crossElasticityEditorOpen} onOpenChange={setCrossElasticityEditorOpen}>
                <DialogContent className="max-w-3xl">
                    <DialogHeader>
                        <DialogTitle>
                            Edit Cross-Elasticity: {editingSourceProduct?.name} → {editingTargetProduct?.name}
                        </DialogTitle>
                    </DialogHeader>
                    {editingRelationship && editingSourceProduct && editingTargetProduct && (
                        <CrossElasticityCurveEditor
                            sourceProduct={editingSourceProduct}
                            targetProduct={editingTargetProduct}
                            relationship={editingRelationship}
                            onSave={handleSaveCrossElasticity}
                            onClose={() => setCrossElasticityEditorOpen(false)}
                        />
                    )}
                </DialogContent>
            </Dialog>
        </div>
    );
}