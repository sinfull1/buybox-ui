"use client"

import React, {useEffect, useState} from "react";
import {Card, CardContent, CardHeader, CardTitle} from "@/components/ui/card";
import {Progress} from "@/components/ui/progress";
import {PieChart, Pie, Cell, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer} from "recharts";
import {Button} from "@/components/ui/button";
import {Input} from "@/components/ui/input";
import {Search} from "lucide-react";

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


const riskData: RiskData[] = [
    {name: "Critical Risk", value: 3, color: "#ff4d4f"},
    {name: "High Risk", value: 12, color: "#ff7a45"},
    {name: "Medium Risk", value: 39, color: "#ffa940"},
    {name: "Low Risk", value: 46, color: "#73d13d"}
];

const heatmapData: HeatmapData[] = [
    {category: "Severe", Rare: 40, Unlikely: 50, Moderate: 40, Likely: 2, Certain: 3},
    {category: "Major", Rare: 50, Unlikely: 50, Moderate: 50, Likely: 3, Certain: 3},
    {category: "Moderate", Rare: 50, Unlikely: 108, Moderate: 150, Likely: 160, Certain: 104},
    {category: "Minor", Rare: 140, Unlikely: 207, Moderate: 101, Likely: 90, Certain: 80},
    {category: "Insignificant", Rare: 200, Unlikely: 140, Moderate: 90, Likely: 80, Certain: 20}
];

const summaryMetrics: SummaryMetric[] = [
    {title: "Recommendations", value: "4"},
    {title: "Concerns ", value: "391"},
    {title: "Catalogue", value: "81"},
    {title: "Order History", value: "213"}
];

let debounceTimeout: string | number | NodeJS.Timeout | undefined;
export default function BuyBoxScore({ /*seller, product, location, score, price, rank */}) {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [showDropdown, setShowDropdown] = useState(false);


    useEffect(() => {
        clearTimeout(debounceTimeout);
        if (query.length > 2) {
            debounceTimeout = setTimeout(() => {
                fetch(`http://localhost:8080/buybox/seller?query=${query}`)
                    .then(response => response.json())
                    .then(data => {
                        setResults(data);
                        setShowDropdown(true);
                    })
                    .catch(error => console.error('Error fetching data:', error));
            }, 200);
        } else {
            setResults([]);
            setShowDropdown(false);
        }

        return () => clearTimeout(debounceTimeout);
    }, [query]);

    const handleSearch = (e) => {
        setQuery(e.target.value);
    };

    const handleSelect = (item) => {
        setQuery(item);
        setShowDropdown(false);
    };

    return (
        <div className="grid grid-cols-2 gap-4 p-6">
            <div className="flex w-full items-center space-x-2">
                <div className="relative w-full max-w-md">
                    <Search className="absolute left-3 top-2.5 text-gray-500" size={20}/>
                    <Input
                        type="text"
                        placeholder="Search Seller..."
                        value={query}
                        onChange={handleSearch}
                        className="pl-10 py-2 text-sm rounded-lg shadow-md focus:ring-2 focus:ring-primary"
                    />
                    {showDropdown && (
                        <ul className="absolute w-full bg-white shadow-md rounded-lg mt-1 max-h-40 overflow-y-auto z-10">
                            {results.map((item, index) => (
                                <li
                                    key={index}
                                    className="px-4 py-2 cursor-pointer hover:bg-gray-100"
                                    onClick={() => handleSelect(item)}
                                >
                                    {item}
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
                <Button type="submit"> Get Details</Button>

            </div>

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
                    <Progress value={87.5} max={100} className="h-3"/>
                    <p className="text-lg font-semibold mt-2">87.5%</p>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle className="text-lg">Response Progress</CardTitle>
                </CardHeader>
                <CardContent>
                    <Progress value={56.2} max={100} className="h-3"/>
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
                                <Cell key={`cell-${index}`} fill={entry.color}/>
                            ))}
                        </Pie>
                        <Tooltip/>
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
                            <CartesianGrid strokeDasharray="3 3"/>
                            <XAxis type="number"/>
                            <YAxis dataKey="category" type="category"/>
                            <Tooltip/>
                            <Bar dataKey="Rare" stackId="a" fill="#f6c23e"/>
                            <Bar dataKey="Unlikely" stackId="a" fill="#e74a3b"/>
                            <Bar dataKey="Moderate" stackId="a" fill="#4e73df"/>
                            <Bar dataKey="Likely" stackId="a" fill="#1cc88a"/>
                            <Bar dataKey="Certain" stackId="a" fill="#36b9cc"/>
                        </BarChart>
                    </ResponsiveContainer>
                </CardContent>
            </Card>
        </div>
    );
};


