import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

interface Props {
  trace: string[];
}

export default function AgentTrace({ trace }: Props) {
  return (
    <Accordion type="single" collapsible className="mt-2">
      <AccordionItem value="trace" className="border-none">
        <AccordionTrigger className="py-2 text-xs text-muted-foreground hover:no-underline">
          Agent trace ({trace.length} steps)
        </AccordionTrigger>
        <AccordionContent>
          <div className="space-y-1">
            {trace.map((step, i) => (
              <div
                key={i}
                className="flex items-start gap-2 text-xs text-muted-foreground"
              >
                <span className="font-mono shrink-0">{i + 1}.</span>
                <span>{step}</span>
              </div>
            ))}
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
