"use client"

import {Card, CardContent} from "@/components/ui/card";
import {Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis} from "recharts";
import React from "react";
import {BuyBoxSummary, PriceChartData} from "@/app/product-dashboard/page";

export interface PriceMonitorProps {
    summary: BuyBoxSummary | undefined,
    data: PriceChartData[] | undefined
}


export default function PriceMonitor(props: PriceMonitorProps) {

    return (<>
        <div className="grid grid-cols-3 gap-4">
            <Card className="m-0">
                <CardContent className="text-center p-0">
                    <p className="text-lg font-bold">{props.summary?.sellerList.length}</p>
                    <p>Sellers</p>
                </CardContent>
            </Card>
            <Card>
                <CardContent className="text-center">
                    <p className="text-lg font-bold">{props.summary?.recPrice}</p>
                    <p>Recommended Retail Price (RRP)</p>
                </CardContent>
            </Card>
            <Card>
                <CardContent className="text-center">
                    <p className="text-lg font-bold">{props.summary?.minPrice} - {props.summary?.maxPrice}</p>
                    <p>Price Range</p>
                </CardContent>
            </Card>
        </div>
        <Card>
            <CardContent>
                <h3 className="text-xl font-bold mb-2">Price History</h3>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={props.data}>
                        <XAxis dataKey="date"/>
                        <YAxis domain={[props.summary?.minPrice, props.summary?.maxPrice]}/>
                        <Tooltip/>
                        <Legend/>
                        {props.summary?.sellerList.map((seller: never, index: number) => (
                            <Line key={seller} type="bumpX" dataKey={seller}
                                  stroke={`hsl(${(index * 137) % 360}, 70%, 50%)`} strokeWidth={1}
                                  dot={false}/>
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </CardContent>
        </Card>
    </>)
}