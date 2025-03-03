"use client"

import React, {useState} from "react";
import {Button} from "@/components/ui/button";

import SearchComponent from "@/app/product-dashboard/search";
import PriceMonitor from "@/app/product-dashboard/price-monitor";

export interface BuyBoxSummary {
    sellerList: [];
    maxPrice: number;
    minPrice: number;
    recPrice: number;
}

export interface PriceChartData {
    date: string;
    seller: string;
    price: number;
}

export default function Page() {
    const [selectedProduct, setSelectedProduct] = useState<string | null>(null);
    const [selectedLocation, setSelectedLocation] = useState<string | null>(null);
    const [data, setData] = useState<PriceChartData[]>();
    const [showDetails, setShowDetails] = useState<boolean | null>(false);
    const [summary, setSummary] = useState<BuyBoxSummary>();
    const [activeTab, setActiveTab] = useState<string>("priceMonitor");

    const fetchOffers = async () => {
        try {
            if (selectedProduct !== null && selectedLocation !== null) {
                const response = await fetch(`http://localhost:8080/buybox/offers/${selectedProduct}/${selectedLocation}`);
                const rawData = await response.json();
                setData(rawData);
                fetchSummary();
                setShowDetails(true);
            }
        } catch (error) {
            console.error("Error fetching offers:", error);
        }
    };

    const fetchSummary = async () => {
        try {
            if (selectedProduct !== null && selectedLocation !== null) {
                const response = await fetch(`http://localhost:8080/buybox/offers/summary/${selectedProduct}/${selectedLocation}`);
                const rawData = await response.json();
                setSummary(rawData);
            }
        } catch (error) {
            console.error("Error fetching offers:", error);
        }
    };

    return (
        <div className="p-6 font-arial space-y-5">
            <h1 className="text-xl font-bold">Pick Product and Location</h1>
            <div className="flex w-full items-center space-x-2">
                <SearchComponent placeholder={"Search Products.."} setter={setSelectedProduct} api={"product"}/>
                <SearchComponent placeholder={"Search Locations.."} setter={setSelectedLocation} api={"location"}/>
                <Button type="submit" onClick={fetchOffers}> Get Details</Button>
            </div>

            <div className="border-b mb-4">
                <nav className="flex space-x-4">
                    <button
                        className={`py-2 px-4 ${activeTab === "priceMonitor" ? "text-blue-600 border-b-2 border-blue-600" : "text-black-600 border-b-2 border-black-600"}`}
                        onClick={() => setActiveTab("priceMonitor")}
                    >
                        Price Monitor
                    </button>
                    <button
                        className={`py-2 px-4 ${activeTab === "saleMonitor" ? "text-blue-600 border-b-2 border-blue-600" : "text-black-600 border-b-2 border-black-600"}`}
                        onClick={() => setActiveTab("saleMonitor")}
                    >
                        Sale Monitor
                    </button>
                </nav>
            </div>

            {showDetails && activeTab === "priceMonitor" && <PriceMonitor summary={summary} data={data}/>}
            {showDetails && activeTab === "saleMonitor" && <PriceMonitor summary={summary} data={data}/>}
        </div>
    );
}