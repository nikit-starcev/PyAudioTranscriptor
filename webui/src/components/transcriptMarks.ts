import { AlertTriangle, ArrowLeftRight, CircleHelp, type LucideIcon } from 'lucide-react'

export type MarkVisual = { Icon: LucideIcon; tone: string }

export const TRANSCRIPT_MARKS: Record<string, MarkVisual> = {
  low_confidence: { Icon: AlertTriangle, tone: 'text-warn' },
  speaker_uncertain: { Icon: CircleHelp, tone: 'text-info' },
  overlap: { Icon: ArrowLeftRight, tone: 'text-primary' },
}
