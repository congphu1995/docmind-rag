import { useCallback, useState } from "react";
import { cn } from "@/lib/utils";
import { Upload, Plus } from "lucide-react";

interface Props {
  onUpload: (file: File) => void;
}

export default function UploadZone({ onUpload }: Props) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) onUpload(file);
    },
    [onUpload]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onUpload(file);
      e.target.value = "";
    },
    [onUpload]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={cn(
        "relative rounded-2xl border-2 border-dashed transition-all duration-300 overflow-hidden",
        isDragging
          ? "border-primary bg-primary/[0.03] scale-[1.01]"
          : "border-border/60 hover:border-primary/30 hover:bg-muted/20"
      )}
    >
      <div className="flex flex-col items-center justify-center py-10 px-6">
        <div
          className={cn(
            "w-12 h-12 rounded-2xl flex items-center justify-center mb-4 transition-colors",
            isDragging ? "bg-primary/10" : "bg-muted/50"
          )}
        >
          <Upload
            className={cn(
              "h-5 w-5 transition-colors",
              isDragging ? "text-primary" : "text-muted-foreground"
            )}
          />
        </div>
        <p className="text-sm font-medium mb-1">
          {isDragging ? "Drop to upload" : "Drop files here"}
        </p>
        <p className="text-xs text-muted-foreground mb-4">
          or browse from your computer
        </p>
        <label className="cursor-pointer inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors">
          <Plus className="h-3.5 w-3.5" />
          Choose file
          <input
            type="file"
            accept=".pdf,.docx,.txt,.md"
            onChange={handleFileSelect}
            className="hidden"
          />
        </label>
        <p className="text-[10px] text-muted-foreground/60 mt-3">
          PDF, DOCX, TXT, MD — up to 50 MB
        </p>
      </div>
    </div>
  );
}
