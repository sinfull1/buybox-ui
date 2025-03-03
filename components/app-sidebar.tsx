import {Calendar, Home, LayoutDashboard} from "lucide-react"

import {
    Sidebar,
    SidebarContent,
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
} from "@/components/ui/sidebar"
import Image from "next/image";

// Menu items.
const items = [
    {
        title: "Seller",
        url: "/seller",
        icon: Home,
    },
    {
        title: "BuyBox",
        url: "/buybox",
        icon: Calendar,
    },
    {
        title: "Seller Dashboard",
        url: "/seller-dashboard",
        icon: LayoutDashboard,
    },
    {
        title: "Product Dashboard",
        url: "/product-dashboard",
        icon: Home,
    },

]

export function AppSidebar() {
    return (
        <Sidebar className="mt-5">
            <SidebarContent className="mt-5">
                <SidebarGroup>
                    <SidebarGroupLabel className="mb-10"> <Image
                        className="dark:invert"
                        src="/tesco-large.png"
                        alt="Tesco logo"
                        width={180}
                        height={38}
                        priority
                    /></SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu className="space-y-2 font-arial">
                            {items.map((item) => (
                                <SidebarMenuItem key={item.title}>
                                    <SidebarMenuButton asChild>
                                        <a href={item.url} className="font-arial">
                                            <item.icon/>
                                            <span>{item.title}</span>
                                        </a>
                                    </SidebarMenuButton>
                                </SidebarMenuItem>
                            ))}
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>
            </SidebarContent>
        </Sidebar>
    )
}
