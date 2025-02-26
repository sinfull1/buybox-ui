"use client"

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { PieChart, Pie, Cell, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts";

type RiskData = {
    name: string;
    value: number;
    color: string;
};

type HeatmapData = {
    category: string;
    Rare: number;
    Unlikely: number;
    Moderate: number;
    Likely: number;
    Certain: number;
};

type SummaryMetric = {
    title: string;
    value: string;
};

type BuyBoxScoreProps = {
    seller: string;
    product: string;
    location: string;
    score: number;
    price: number;
    rank: number;
};

const riskData: RiskData[] = [
    { name: "Critical Risk", value: 3, color: "#ff4d4f" },
    { name: "High Risk", value: 12, color: "#ff7a45" },
    { name: "Medium Risk", value: 39, color: "#ffa940" },
    { name: "Low Risk", value: 46, color: "#73d13d" }
];

const heatmapData: HeatmapData[] = [
    { category: "Severe", Rare: 40, Unlikely: 50, Moderate: 40, Likely: 2, Certain: 3 },
    { category: "Major", Rare: 50, Unlikely: 50, Moderate: 50, Likely: 3, Certain: 3 },
    { category: "Moderate", Rare: 50, Unlikely: 108, Moderate: 150, Likely: 160, Certain: 104 },
    { category: "Minor", Rare: 140, Unlikely: 207, Moderate: 101, Likely: 90, Certain: 80 },
    { category: "Insignificant", Rare: 200, Unlikely: 140, Moderate: 90, Likely: 80, Certain: 20 }
];

const summaryMetrics: SummaryMetric[] = [
    { title: "% Risks", value: "37.5%" },
    { title: "% of Risks", value: "391" },
    { title: "Risk Analysis Progress", value: "87.5%" },
    { title: "Response Progress for Risks", value: "56.2%" }
];

export default function BuyBoxScore ({ /*seller, product, location, score, price, rank */}) {
    return (
        <div className="grid grid-cols-2 gap-4 p-6">
            <Card className="col-span-2 text-center">
                <CardHeader>
                    <CardTitle className="text-2xl font-bold">Risk Management Dashboard</CardTitle>
                </CardHeader>
            </Card>

            <div className="grid grid-cols-4 gap-4 col-span-2">
                {summaryMetrics.map((metric, index) => (
                    <Card key={index} className="text-center p-4">
                        <CardHeader>
                            <CardTitle className="text-lg">{metric.title}</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <p className="text-2xl font-bold">{metric.value}</p>
                        </CardContent>
                    </Card>
                ))}
            </div>

            <Card>
                <CardHeader>
                    <CardTitle className="text-lg">Risk Analysis Progress</CardTitle>
                </CardHeader>
                <CardContent>
                    <Progress value={87.5} max={100} className="h-3" />
                    <p className="text-lg font-semibold mt-2">87.5%</p>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle className="text-lg">Response Progress</CardTitle>
                </CardHeader>
                <CardContent>
                    <Progress value={56.2} max={100} className="h-3" />
                    <p className="text-lg font-semibold mt-2">56.2%</p>
                </CardContent>
            </Card>

            <Card className="col-span-2">
                <CardHeader>
                    <CardTitle className="text-lg">Risk Rating Breakdown</CardTitle>
                </CardHeader>
                <CardContent>
                    <PieChart width={300} height={300}>
                        <Pie data={riskData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label>
                            {riskData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                        </Pie>
                        <Tooltip />
                    </PieChart>
                </CardContent>
            </Card>

            <Card className="col-span-2">
                <CardHeader>
                    <CardTitle className="text-lg">Risk Heat Map</CardTitle>
                </CardHeader>
                <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={heatmapData} layout="vertical">
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis type="number" />
                            <YAxis dataKey="category" type="category" />
                            <Tooltip />
                            <Bar dataKey="Rare" stackId="a" fill="#f6c23e" />
                            <Bar dataKey="Unlikely" stackId="a" fill="#e74a3b" />
                            <Bar dataKey="Moderate" stackId="a" fill="#4e73df" />
                            <Bar dataKey="Likely" stackId="a" fill="#1cc88a" />
                            <Bar dataKey="Certain" stackId="a" fill="#36b9cc" />
                        </BarChart>
                    </ResponsiveContainer>
                </CardContent>
            </Card>
        </div>
    );
};


