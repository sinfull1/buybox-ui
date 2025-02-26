"use client"

import React, {useState} from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {Select, SelectContent, SelectItem, SelectTrigger, SelectValue} from "@/components/ui/select";
import {locationNames, productNames, sellerNames} from "@/app/model";



export default function Page() {
    const [selectedProduct, setSelectedProduct] = useState<string | null>(null);
    const [selectedLocation, setSelectedLocation] = useState<string | null>(null);
    const [data, setData] = useState([
        { date: "18/06/24", seller: "Amazon", price: 100 },
    ]);

    const fetchOffers = async () => {
        try {
            const response = await fetch(`http://localhost:8080/buybox/offers/${selectedProduct}/${selectedLocation}`);
            const latestChartData = await response.json();
            setData(latestChartData);

        } catch (error) {
            console.error("Error fetching offers:", error);
        }
    };



    return (
        <div className="p-6">
            <h1 className="text-xl font-bold">Offers</h1>
            <div className="grid grid-cols-3 gap-4">
                <Select onValueChange={(value) => setSelectedProduct(value)}>
                    <SelectTrigger><SelectValue placeholder="Select Product" /></SelectTrigger>
                    <SelectContent>
                        {productNames.map((product)=> (
                            <SelectItem key={product} value={product}>{product}</SelectItem>
                        ))}
                    </SelectContent>
                </Select>
                <Select onValueChange={(value) => setSelectedLocation(value)}>
                    <SelectTrigger><SelectValue placeholder="Select Location" /></SelectTrigger>
                    <SelectContent>
                        {locationNames.map((location)=> (
                            <SelectItem  key={location} value={location}>{location}</SelectItem>
                        ))}
                    </SelectContent>
                </Select>
                <Button onClick={fetchOffers}>Get Prices</Button>
            </div>
            <h2 className="text-2xl font-bold">{selectedProduct}</h2>
            <div className="border-b mb-4">
                <nav className="flex space-x-4">
                    <button className="py-2 px-4 text-blue-600 border-b-2 border-blue-600">Price Monitor</button>
                    <button className="py-2 px-4">Product Metrics</button>
                    <button className="py-2 px-4">Brand Reputation</button>
                    <button className="py-2 px-4">Content Analysis</button>
                </nav>
            </div>
            <div className="grid grid-cols-3 gap-4 mt-6">
                <Card>
                    <CardContent className="text-center">
                        <p className="text-lg font-bold">{sellerNames.length}</p>
                        <p>Sellers</p>
                    </CardContent>
                </Card>
                <Card>
                    <CardContent className="text-center">
                        <p className="text-lg font-bold">£13.49</p>
                        <p>Recommended Retail Price (RRP)</p>
                    </CardContent>
                </Card>
                <Card>
                    <CardContent className="text-center">
                        <p className="text-lg font-bold">£13.49</p>
                        <p>Price Range</p>
                    </CardContent>
                </Card>
            </div>
            <Card>
                <CardContent>
                    <h3 className="text-xl font-bold mb-2">Price History</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={data}>
                            <XAxis dataKey="date" />
                            <YAxis domain={[data[0].price - data[0].price/4,data[0].price + data[0].price/4 ]}/>
                            <Tooltip />
                            <Legend />
                            {sellerNames.map((seller, index) => (
                                <Line key={seller} type="bump" dataKey={seller} stroke={`hsl(${(index * 137) % 360}, 70%, 50%)`}  strokeWidth={1} dot={false} />
                            ))}
                        </LineChart>
                    </ResponsiveContainer>
                </CardContent>
            </Card>
        </div>
    );
};
