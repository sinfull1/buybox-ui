export const productNames: string [] = [
    "LAPTOP_GAMING",
    "SMARTPHONE_PRO",
    "WIRELESS_HEADPHONES",
    "BLUETOOTH_SPEAKER",
    "MECHANICAL_KEYBOARD",
    "ULTRAWIDE_MONITOR",
    "SMARTWATCH_ELITE",
    "GRAPHICS_CARD",
    "EXTERNAL_SSD",
    "GAMING_MOUSE"
];
export const sellerNames : string [] = [
    "TECH_WORLD",
    "GADGET_HUB",
    "ELECTRO_STORE",
    "MEGA_RETAIL",
    "DIGITAL_MART",
    "SMART_TECH",
    "HYPER_SHOP",
    "INNOVATE_GEAR",
    "FAST_BUY",
    "TRENDY_TECH"
];
export const locationNames: string []  = [
    "NEW_YORK_WAREHOUSE",
    "LOS_ANGELES_HUB",
    "CHICAGO_DISTRIBUTION",
    "HOUSTON_DEPOT",
    "MIAMI_CENTRAL",
    "SAN_FRANCISCO_STORE",
    "SEATTLE_OUTLET",
    "DALLAS_SUPPLY",
    "BOSTON_SHIPPING",
    "ATLANTA_STORAGE"
];

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