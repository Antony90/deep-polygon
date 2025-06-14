import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Brain } from "lucide-react"

interface ParameterProps {
  label: string
  value: string | number
}

function Parameter({ label, value }: ParameterProps) {
  return (
    <div className="flex items-center border-b border-border/30 last:border-0 py-2">
      <div className="w-1/2 text-sm text-muted-foreground">{label}</div>
      <div className="w-1/2 font-mono text-sm truncate">{value}</div>
    </div>
  )
}

interface SectionProps {
  title: string
  parameters: Array<{ label: string; value: string | number }>
}

function Section({ title, parameters }: SectionProps) {
  return (
    <div>
      <h3 className="text-sm font-medium mb-2">{title}</h3>
      <div className="bg-muted/40 rounded-lg p-2.5">
        {parameters.map((param, idx) => (
          <Parameter key={idx} label={param.label} value={param.value} />
        ))}
      </div>
    </div>
  )
}

export function MLOverviewCard() {
  // Define all parameters in a structured way
  const sections = [
    {
      title: "Learning Parameters",
      parameters: [
        { label: "Learning Rate", value: "0.0003" },
        { label: "Gamma", value: "0.95" },
        { label: "Batch Size", value: "128" },
        { label: "Memory Size", value: "100,000" },
      ],
    },
    {
      title: "Exploration Parameters",
      parameters: [
        { label: "Epsilon Decay Steps", value: "500,000" },
        { label: "Epsilon Min", value: "0.1" },
        { label: "Update Target Rate", value: "5,000" },
        { label: "Map Size", value: "100 x 100" },
      ],
    },
    {
      title: "Model Architecture",
      parameters: [
        { label: "Parameters", value: "1,256,324" },
        { label: "Input Shape", value: "[1, 84, 84, 4]" },
        { label: "Output Shape", value: "[1, 6]" },
        { label: "Model Size", value: "4.8 MB" },
      ],
    },
    {
      title: "Training Configuration",
      parameters: [
        { label: "Optimizer", value: "Adam" },
        { label: "Loss Function", value: "Huber Loss" },
        { label: "Training Device", value: "CUDA" },
      ],
    },
  ]

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Brain className="size-5" />
          <CardTitle className="text-lg font-bold">Machine Learning Configuration</CardTitle>
        </div>
      </CardHeader>

      <CardContent>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {sections.map((section, idx) => (
            <Section key={idx} title={section.title} parameters={section.parameters} />
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
