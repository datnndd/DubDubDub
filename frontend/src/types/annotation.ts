export interface AnnotationStyle {
  bgColor: string;
  bgOpacity: number;
  textColor: string;
  fontSize?: number;
  bold?: boolean;
  fontFamily?: string;
}

export interface AnnotationItem {
  id: string;
  startSec: number;
  endSec: number;
  startTime?: string;
  endTime?: string;
  box: [number, number, number, number];
  sourceText: string;
  targetText: string;
  confidence?: number;
  style: AnnotationStyle;
}

export interface RenderedVideoRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export function computeRenderedVideoRect(
  containerWidth: number,
  containerHeight: number,
  videoWidth: number,
  videoHeight: number
): RenderedVideoRect {
  if (
    !Number.isFinite(containerWidth) ||
    !Number.isFinite(containerHeight) ||
    containerWidth <= 0 ||
    containerHeight <= 0
  ) {
    return { left: 0, top: 0, width: 0, height: 0 };
  }
  if (
    !Number.isFinite(videoWidth) ||
    !Number.isFinite(videoHeight) ||
    videoWidth <= 0 ||
    videoHeight <= 0
  ) {
    return { left: 0, top: 0, width: containerWidth, height: containerHeight };
  }

  const containerAspect = containerWidth / containerHeight;
  const videoAspect = videoWidth / videoHeight;

  if (Math.abs(containerAspect - videoAspect) < 1e-4) {
    return { left: 0, top: 0, width: containerWidth, height: containerHeight };
  }

  if (videoAspect > containerAspect) {
    const renderedHeight = containerWidth / videoAspect;
    return {
      left: 0,
      top: (containerHeight - renderedHeight) / 2,
      width: containerWidth,
      height: renderedHeight,
    };
  }

  const renderedWidth = containerHeight * videoAspect;
  return {
    left: (containerWidth - renderedWidth) / 2,
    top: 0,
    width: renderedWidth,
    height: containerHeight,
  };
}

function hexToAssBgr(hex: string, fallback = '1B1818'): string {
  const m = /^#?([0-9a-fA-F]{6})$/.exec(String(hex || '').trim());
  if (!m) return `&H${fallback}&`;
  const h = m[1].toUpperCase();
  return `&H${h.slice(4, 6)}${h.slice(2, 4)}${h.slice(0, 2)}&`;
}

function formatAssTime(sec: number): string {
  const totalCs = Math.max(0, Math.round(Number(sec || 0) * 100));
  const cs = totalCs % 100;
  const totalSec = Math.floor(totalCs / 100);
  const s = totalSec % 60;
  const totalMin = Math.floor(totalSec / 60);
  const m = totalMin % 60;
  const h = Math.floor(totalMin / 60);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${h}:${pad(m)}:${pad(s)}.${pad(cs)}`;
}

export function generateAnnotationsAssContent(
  annotations: AnnotationItem[],
  width = 1920,
  height = 1080
): string {
  const playW = Math.max(64, Math.round(width || 1920));
  const playH = Math.max(64, Math.round(height || 1080));
  const sorted = [...(annotations || [])]
    .filter((a) => Boolean((a?.targetText || a?.sourceText || '').trim()))
    .sort((a, b) => a.startSec - b.startSec || (a.box?.[1] ?? 0) - (b.box?.[1] ?? 0));

  const styles: string[] = [
    'Style: AnnBox,Arial,20,&H00000000&,&H00000000&,&H00000000&,&H00000000&,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1',
  ];
  const events: string[] = [];

  sorted.forEach((ann, idx) => {
    const seq = idx + 1;
    const [bx, by, bw, bh] = ann.box || [0.1, 0.1, 0.3, 0.08];
    const x0 = Math.max(0, Math.round(bx * playW));
    const y0 = Math.max(0, Math.round(by * playH));
    const wPx = Math.max(12, Math.round(bw * playW));
    const hPx = Math.max(12, Math.round(bh * playH));
    const cx = (x0 + wPx / 2).toFixed(1);
    const cy = (y0 + hPx / 2).toFixed(1);

    const style = ann.style || { bgColor: '#18181B', bgOpacity: 92, textColor: '#FFFFFF', bold: true };
    const fs = Math.max(12, Math.round(style.fontSize || hPx * 0.72));
    const textCol = `&H00${hexToAssBgr(style.textColor || '#FFFFFF', 'FFFFFF').slice(2)}`;
    const boldFlag = style.bold === false ? 0 : -1;
    const fontName = String(style.fontFamily || 'Arial').replace(/,/g, ' ');
    styles.push(
      `Style: AnnText_${seq},${fontName},${fs},${textCol},${textCol},&H60000000&,&H00000000&,${boldFlag},0,0,0,100,100,0,0,1,1,0,5,6,6,6,1`
    );

    const startAss = formatAssTime(ann.startSec);
    const endAss = formatAssTime(Math.max(ann.startSec + 0.1, ann.endSec));
    const srcClean = String(ann.sourceText || '').replace(/[\r\n]+/g, ' / ');
    events.push(
      `Comment: 0,${startAss},${endAss},AnnBox,,0,0,0,,[SRC] ${srcClean} | id=${ann.id} | box=${bx.toFixed(4)},${by.toFixed(4)},${bw.toFixed(4)},${bh.toFixed(4)}`
    );

    const opacity = Math.max(0, Math.min(100, Number(style.bgOpacity ?? 92)));
    if (opacity > 0.5) {
      const alphaHex = Math.round((1 - opacity / 100) * 255)
        .toString(16)
        .toUpperCase()
        .padStart(2, '0');
      const boxBgr = hexToAssBgr(style.bgColor || '#18181B', '1B1818');
      events.push(
        `Dialogue: 0,${startAss},${endAss},AnnBox,,0,0,0,,{\\an7\\pos(${x0},${y0})\\1c${boxBgr}\\1a&H${alphaHex}&\\bord0\\shad0\\p1}m 0 0 l ${wPx} 0 l ${wPx} ${hPx} l 0 ${hPx}{\\p0}`
      );
    }

    const display = String(ann.targetText || ann.sourceText || '')
      .replace(/\\/g, '\\\\')
      .replace(/\{/g, '\\{')
      .replace(/\}/g, '\\}')
      .replace(/\r?\n/g, '\\N');
    events.push(
      `Dialogue: 1,${startAss},${endAss},AnnText_${seq},,0,0,0,,{\\an5\\pos(${cx},${cy})}${display}`
    );
  });

  return [
    '[Script Info]',
    '; Generated by DubDub Studio — On-Screen Annotations',
    'ScriptType: v4.00+',
    `PlayResX: ${playW}`,
    `PlayResY: ${playH}`,
    'WrapStyle: 0',
    'ScaledBorderAndShadow: yes',
    '',
    '[V4+ Styles]',
    'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
    ...styles,
    '',
    '[Events]',
    'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text',
    ...events,
    '',
  ].join('\n');
}
