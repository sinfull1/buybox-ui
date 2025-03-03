"use client";

import React, {useState} from "react";
import {Card, CardContent} from "@/components/ui/card";
import {Button} from "@/components/ui/button";
import {BuyBoxOffer} from "../model";
import SearchComponent from "@/app/product-dashboard/search";


export default function Page() {
    const [selectedProduct, setSelectedProduct] = useState<string | null>(null);
    const [selectedLocation, setSelectedLocation] = useState<string | null>(null);
    const [offers, setOffers] = useState<BuyBoxOffer[]>([]);

    const fetchOffers = async () => {
        try {
            const response = await fetch(`http://localhost:8080/buybox/winner/${selectedProduct}/${selectedLocation}`);
            const data: BuyBoxOffer[] = await response.json();
            setOffers(data);
        } catch (error) {
            console.error("Error fetching offers:", error);
        }
    };

    return (
        <div className="p-6 w-full mx-auto space-y-4">
            <h1 className="text-xl font-bold">Current Buybox Winner By Product & Location</h1>
            <div className="flex w-full items-center space-x-2">
                <SearchComponent placeholder={"Search Products.."} setter={setSelectedProduct} api={"product"}/>
                <SearchComponent placeholder={"Search Locations.."} setter={setSelectedLocation} api={"location"}/>
                <Button type="submit" onClick={fetchOffers}> Get Offers</Button>

            </div>
            <div className="flex flex-row justify-evenly flex-wrap gap-4 p-4 bg-gray-100">
                {offers.length > 0 ? (
                    offers.map((offer) => (
                        <Card key={offer.buyBoxOfferKey.effectiveAt}>
                            <CardContent className="p-4">
                                <p><strong>Product:</strong> {offer.buyBoxOfferKey.productId}</p>
                                <p><strong>Seller:</strong> {offer.buyBoxOfferKey.sellerId}</p>
                                <p><strong>Location:</strong> {offer.buyBoxOfferKey.locationId}</p>
                                <p><strong>Price:</strong> {offer.price}</p>
                            </CardContent>
                        </Card>
                    ))
                ) : (
                    <p className="text-gray-500">No offers available.</p>
                )}
            </div>
        </div>
    );
}

