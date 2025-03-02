import React, {useEffect, useState} from "react";
import {Input} from "@/components/ui/input";
import {Search} from "lucide-react";





interface SearchComponentProps {
    placeholder: string
    setter: any
    api: string
}

export default function SearchComponent(props: SearchComponentProps) {
    let debounceTimeout: string | number | NodeJS.Timeout | undefined;
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [showDropdown, setShowDropdown] = useState(false);

    useEffect(() => {
        clearTimeout(debounceTimeout);
        if (query.length > 2) {
            debounceTimeout = setTimeout(() => {
                fetch(`http://localhost:8080/buybox/`+props.api+`/${query}`)
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


    const handleSearch = (e: { target: { value: React.SetStateAction<string>; }; }) => {
        setQuery(e.target.value);
    };

    const handleSelect = (item: React.SetStateAction<string>) => {
        setQuery(item);
        setShowDropdown(false);
        props.setter(item);
    };


    return (
        <div className="relative w-full max-w-md">
            <Search className="absolute left-3 top-2.5 text-gray-500" size={20}/>
            <Input
                type="text"
                placeholder= {props.placeholder}
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
    )
}