import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface Props {
  value: "openai" | "claude";
  onChange: (v: "openai" | "claude") => void;
}

export default function LLMToggle({ value, onChange }: Props) {
  return (
    <Tabs value={value} onValueChange={(v) => onChange(v as "openai" | "claude")}>
      <TabsList className="h-8">
        <TabsTrigger value="openai" className="text-xs px-3">
          GPT-4o
        </TabsTrigger>
        <TabsTrigger value="claude" className="text-xs px-3">
          Claude
        </TabsTrigger>
      </TabsList>
    </Tabs>
  );
}
