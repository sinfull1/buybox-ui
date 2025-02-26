"use client";

import { useState } from "react";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { locationNames, productNames} from "../model";

export interface BuyBoxOfferKey {
    productId: string;
    sellerId: string;
    locationId: string;
    effectiveAt: string; // Using string to represent LocalDateTime
    // Using number to represent BigDecimal
}

export interface BuyBoxOffer {
    buyBoxOfferKey: BuyBoxOfferKey;
    price: number;
    lastUpdated: string; // Using string to represent LocalDateTime
    tags: Record<string, string>;
}
export default function Page() {
    const [selectedProduct, setSelectedProduct] = useState<string | null>(null);
    const [selectedLocation, setSelectedLocation] = useState<string | null>(null);
    const [offers, setOffers] = useState<BuyBoxOffer[]>([]);

    const fetchOffers = async () => {
        try {
            const response = await fetch(`http://localhost:8080/events/winner/${selectedProduct}/${selectedLocation}`);
            const data: BuyBoxOffer[] = await response.json();
            setOffers(data);
        } catch (error) {
            console.error("Error fetching offers:", error);
        }
    };

    return (
        <div className="p-6 max-w-xl mx-auto space-y-4">
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
            </div>

            <Button onClick={fetchOffers}>Get Offers</Button>

            <div className="space-y-2">
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

