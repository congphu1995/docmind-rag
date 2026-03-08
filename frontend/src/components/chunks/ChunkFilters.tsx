import { Search, Filter, FileText } from "lucide-react";
import { useChunkStore } from "@/stores/chunkStore";

const TYPE_OPTIONS = ["All", "Text", "Table", "Figure", "Code"] as const;

export default function ChunkFilters() {
  const { filters, setFilters } = useChunkStore();

  return (
    <div className="flex items-center gap-3 flex-wrap">
      {/* Type dropdown */}
      <div className="relative">
        <Filter className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
        <select
          value={filters.type ?? "All"}
          onChange={(e) =>
            setFilters({
              type: e.target.value === "All" ? undefined : e.target.value.toLowerCase(),
            })
          }
          className="pl-9 pr-3 py-2 rounded-xl border border-border/50 bg-background/50 text-sm outline-none focus:border-primary/40 transition-colors appearance-none cursor-pointer min-w-[120px]"
        >
          {TYPE_OPTIONS.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </div>

      {/* Page number input */}
      <div className="relative">
        <FileText className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
        <input
          type="number"
          min={1}
          placeholder="Page"
          value={filters.page ?? ""}
          onChange={(e) =>
            setFilters({
              page: e.target.value ? Number(e.target.value) : undefined,
            })
          }
          className="pl-9 pr-3 py-2 rounded-xl border border-border/50 bg-background/50 text-sm outline-none focus:border-primary/40 transition-colors w-24 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
        />
      </div>

      {/* Search text input */}
      <div className="relative flex-1 min-w-[200px]">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
        <input
          type="text"
          placeholder="Search chunk content..."
          value={filters.search ?? ""}
          onChange={(e) =>
            setFilters({ search: e.target.value || undefined })
          }
          className="w-full pl-9 pr-3 py-2 rounded-xl border border-border/50 bg-background/50 text-sm outline-none focus:border-primary/40 transition-colors"
        />
      </div>
    </div>
  );
}
