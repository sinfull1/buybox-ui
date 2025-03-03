"use client";

import React, {useState} from "react";
import {Card, CardContent} from "@/components/ui/card";
import {Button} from "@/components/ui/button";
import {BuyBoxOffer} from "../model";
import SearchComponent from "@/app/product-dashboard/search";


function uuidv4() {
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
        (+c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> +c / 4).toString(16)
    );
}

export default function Page() {
    const [selectedSeller, setSelectedSeller] = useState<string | null>(null);
    const [selectedLocation, setSelectedLocation] = useState<string | null>(null);
    const [offers, setOffers] = useState<BuyBoxOffer[]>([]);

    const fetchOffers = async () => {
        try {
            console.log(selectedSeller);
            const response = await fetch(`http://localhost:8080/buybox/seller/${selectedSeller}/${selectedLocation}`);
            const data: BuyBoxOffer[] = await response.json();
            setOffers(data);
        } catch (error) {
            console.error("Error fetching offers:", error);
        }
    };

    return (
        <div className="p-6 mx-auto space-y-4">
            <h1 className="text-xl font-bold">Get Seller Catalogue</h1>
            <div className="flex w-full items-center space-x-2">
                <SearchComponent placeholder={"Search Seller.."} setter={setSelectedSeller} api={"seller"}/>
                <SearchComponent placeholder={"Search Locations.."} setter={setSelectedLocation} api={"location"}/>
                <Button type="submit" onClick={fetchOffers}>Get Offers</Button>

            </div>


            <div className="flex flex-row justify-evenly flex-wrap gap-4 p-4 bg-gray-100">
                {offers.length > 0 ? (
                    offers.map((offer) => (
                        <Card key={uuidv4()}>
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

