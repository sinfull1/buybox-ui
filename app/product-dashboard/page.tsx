"use client"

import React, { useState} from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
//import {Select, SelectContent, SelectItem, SelectTrigger, SelectValue} from "@/components/ui/select";
import {  sellerNames} from "@/app/model";

import SearchComponent from "@/app/product-dashboard/search";


export default function Page() {
    const [selectedProduct, setSelectedProduct] = useState<string | null>(null);
    const [selectedLocation, setSelectedLocation] = useState<string | null>(null);
    const [data, setData] = useState([
        { date: "1", seller: "1", price:0  },
    ]);

    const fetchOffers = async () => {
        try {
            if (selectedProduct !== null && selectedLocation !== null  ) {
                const response = await fetch(`http://localhost:8080/buybox/offers/${selectedProduct}/${selectedLocation}`);
                const latestChartData = await response.json();
                setData(latestChartData);
            }

        } catch (error) {
            console.error("Error fetching offers:", error);
        }
    };



    return (
        <div className="p-6 font-arial space-y-5">
            <h1 className="text-xl font-bold">Pick Product and Location</h1>
            <div className="flex w-full items-center space-x-2">
                <SearchComponent placeholder={"Search Products.."}  setter={setSelectedProduct} api={"product"}/>
                <SearchComponent placeholder={"Search Locations.."} setter={setSelectedLocation} api={"location"}/>
                <Button type="submit" onClick={fetchOffers}> Get Details</Button>

            </div>

            <div className="border-b mb-4">
                <nav className="flex space-x-4">
                    <button className="py-2 px-4 text-blue-600 border-b-2 border-blue-600">Price Monitor</button>
                   </nav>
            </div>
            <div className="grid grid-cols-3 gap-4">
                <Card className="m-0">
                    <CardContent className="text-center p-0">
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
