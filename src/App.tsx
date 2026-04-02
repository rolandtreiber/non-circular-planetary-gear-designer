import { useEffect, useRef, useState } from 'react'
import './App.css'

type Point = {
  x: number
  y: number
}

type TurningCandidate = Point & {
  score: number
}

const TAU = Math.PI * 2
const VIEWBOX_SIZE = 880
const EDITOR_SIZE = 320
const CENTER = VIEWBOX_SIZE / 2
const EDITOR_CENTER = EDITOR_SIZE / 2
const DEFAULT_RADII = [154, 170, 148, 188, 134, 178, 144, 186, 146, 174, 138, 180]

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function polarToCartesian(angle: number, radius: number): Point {
  return {
    x: Math.cos(angle) * radius,
    y: Math.sin(angle) * radius,
  }
}

function distance(a: Point, b: Point) {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function normalize(point: Point) {
  const length = Math.hypot(point.x, point.y) || 1
  return { x: point.x / length, y: point.y / length }
}

function rotatePoint(point: Point, angle: number): Point {
  const cos = Math.cos(angle)
  const sin = Math.sin(angle)
  return {
    x: point.x * cos - point.y * sin,
    y: point.x * sin + point.y * cos,
  }
}

function scalePoint(point: Point, scale: number): Point {
  return {
    x: point.x * scale,
    y: point.y * scale,
  }
}

function sampleClosedCatmullRom(points: Point[], samplesPerSegment: number) {
  const samples: Point[] = []

  for (let i = 0; i < points.length; i += 1) {
    const p0 = points[(i - 1 + points.length) % points.length]
    const p1 = points[i]
    const p2 = points[(i + 1) % points.length]
    const p3 = points[(i + 2) % points.length]

    for (let step = 0; step < samplesPerSegment; step += 1) {
      const t = step / samplesPerSegment
      const t2 = t * t
      const t3 = t2 * t
      samples.push({
        x:
          0.5 *
          ((2 * p1.x) +
            (-p0.x + p2.x) * t +
            (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 +
            (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3),
        y:
          0.5 *
          ((2 * p1.y) +
            (-p0.y + p2.y) * t +
            (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 +
            (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3),
      })
    }
  }

  return samples
}

function createPitchCurve(controlRadii: number[], radialBias: number, smoothing: number) {
  const controlPoints = controlRadii.map((radius, index) => {
    const angle = (index / controlRadii.length) * TAU
    const harmonic = Math.cos(angle * 3) * radialBias + Math.sin(angle * 2) * radialBias * 0.45
    return polarToCartesian(angle, radius + harmonic)
  })

  const samplesPerSegment = Math.round(clamp(12 + smoothing * 0.28, 8, 28))
  return sampleClosedCatmullRom(controlPoints, samplesPerSegment)
}

function resampleClosedPolyline(points: Point[], sampleCount: number) {
  if (points.length === 0 || sampleCount <= 0) {
    return []
  }

  const { cumulative, total } = computeArcLengths(points)
  const samples: Point[] = []

  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
    const target = (sampleIndex / sampleCount) * total
    let segmentIndex = 0

    while (segmentIndex < points.length - 1 && cumulative[segmentIndex + 1] < target) {
      segmentIndex += 1
    }

    const segmentStart = points[segmentIndex]
    const segmentEnd = points[(segmentIndex + 1) % points.length]
    const segmentLength = cumulative[segmentIndex + 1] - cumulative[segmentIndex] || 1
    const localT = (target - cumulative[segmentIndex]) / segmentLength

    samples.push({
      x: segmentStart.x + (segmentEnd.x - segmentStart.x) * localT,
      y: segmentStart.y + (segmentEnd.y - segmentStart.y) * localT,
    })
  }

  return samples
}

function computeArcLengths(points: Point[]) {
  const cumulative = [0]
  let total = 0

  for (let i = 0; i < points.length; i += 1) {
    total += distance(points[i], points[(i + 1) % points.length])
    cumulative.push(total)
  }

  return { cumulative, total }
}

function computeSignedArea(points: Point[]) {
  let area = 0
  for (let i = 0; i < points.length; i += 1) {
    const current = points[i]
    const next = points[(i + 1) % points.length]
    area += current.x * next.y - next.x * current.y
  }
  return area / 2
}

function createToothedOutline(
  points: Point[],
  toothCount: number,
  toothDepth: number,
  toothSharpness: number,
  invert = false,
) {
  const { cumulative, total } = computeArcLengths(points)
  const orientation = computeSignedArea(points) >= 0 ? 1 : -1
  const harmonic = 0.5 + toothSharpness * 0.04
  const addendum = toothDepth * 0.46
  const dedendum = toothDepth * 0.42
  const blend = toothDepth * 0.08

  return points.map((point, index) => {
    const prev = points[(index - 1 + points.length) % points.length]
    const next = points[(index + 1) % points.length]
    const tangent = normalize({ x: next.x - prev.x, y: next.y - prev.y })
    const outward = normalize({
      x: tangent.y * orientation,
      y: -tangent.x * orientation,
    })
    const phase = (cumulative[index] / total) * toothCount * TAU
    const wave = Math.cos(phase)
    const smoothWave = Math.sin(phase) * Math.sin(phase)
    const roundedCrest = Math.sign(wave) * Math.pow(Math.abs(wave), harmonic)
    const offset =
      roundedCrest >= 0
        ? addendum * roundedCrest + blend * smoothWave
        : dedendum * roundedCrest - blend * smoothWave * 0.65
    const signedOffset = invert ? -offset : offset

    return {
      x: point.x + outward.x * signedOffset,
      y: point.y + outward.y * signedOffset,
    }
  })
}

function pointInPolygon(point: Point, polygon: Point[]) {
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const a = polygon[i]
    const b = polygon[j]
    const intersects =
      a.y > point.y !== b.y > point.y &&
      point.x < ((b.x - a.x) * (point.y - a.y)) / ((b.y - a.y) || 1e-6) + a.x
    if (intersects) {
      inside = !inside
    }
  }
  return inside
}

function distancePointToSegment(point: Point, start: Point, end: Point) {
  const segment = { x: end.x - start.x, y: end.y - start.y }
  const lengthSquared = segment.x * segment.x + segment.y * segment.y || 1
  const t = clamp(
    ((point.x - start.x) * segment.x + (point.y - start.y) * segment.y) / lengthSquared,
    0,
    1,
  )
  const projected = {
    x: start.x + segment.x * t,
    y: start.y + segment.y * t,
  }
  return distance(point, projected)
}

function findTurningCandidates(points: Point[], minInset: number) {
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity

  points.forEach((point) => {
    minX = Math.min(minX, point.x)
    maxX = Math.max(maxX, point.x)
    minY = Math.min(minY, point.y)
    maxY = Math.max(maxY, point.y)
  })

  const candidates: TurningCandidate[] = []
  const grid = 30
  const stepX = (maxX - minX) / grid
  const stepY = (maxY - minY) / grid

  for (let xIndex = 0; xIndex <= grid; xIndex += 1) {
    for (let yIndex = 0; yIndex <= grid; yIndex += 1) {
      const point = {
        x: minX + xIndex * stepX,
        y: minY + yIndex * stepY,
      }
      if (!pointInPolygon(point, points)) {
        continue
      }

      let minDistance = Infinity
      for (let i = 0; i < points.length; i += 1) {
        minDistance = Math.min(
          minDistance,
          distancePointToSegment(point, points[i], points[(i + 1) % points.length]),
        )
      }

      if (minDistance >= minInset) {
        candidates.push({ ...point, score: minDistance })
      }
    }
  }

  candidates.sort((a, b) => b.score - a.score)
  const filtered: TurningCandidate[] = []

  candidates.forEach((candidate) => {
    if (filtered.length >= 7) {
      return
    }
    const tooClose = filtered.some((kept) => distance(candidate, kept) < candidate.score * 0.75)
    if (!tooClose) {
      filtered.push(candidate)
    }
  })

  return filtered
}

function pointsToPath(points: Point[], cx: number, cy: number) {
  if (points.length === 0) {
    return ''
  }
  return points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x + cx} ${point.y + cy}`)
    .join(' ')
    .concat(' Z')
}

function pointsToSmoothPath(points: Point[], cx: number, cy: number) {
  if (points.length === 0) {
    return ''
  }
  if (points.length < 3) {
    return pointsToPath(points, cx, cy)
  }

  let path = `M ${points[0].x + cx} ${points[0].y + cy}`

  for (let index = 0; index < points.length; index += 1) {
    const p0 = points[(index - 1 + points.length) % points.length]
    const p1 = points[index]
    const p2 = points[(index + 1) % points.length]
    const p3 = points[(index + 2) % points.length]
    const cp1 = {
      x: p1.x + (p2.x - p0.x) / 6,
      y: p1.y + (p2.y - p0.y) / 6,
    }
    const cp2 = {
      x: p2.x - (p3.x - p1.x) / 6,
      y: p2.y - (p3.y - p1.y) / 6,
    }

    path += ` C ${cp1.x + cx} ${cp1.y + cy}, ${cp2.x + cx} ${cp2.y + cy}, ${p2.x + cx} ${p2.y + cy}`
  }

  return `${path} Z`
}

function translatePath(points: Point[], center: Point) {
  return points.map((point) => ({
    x: point.x + center.x,
    y: point.y + center.y,
  }))
}

function transformGear(points: Point[], center: Point, rotation: number) {
  return points.map((point) => {
    const rotated = rotatePoint(point, rotation)
    return {
      x: rotated.x + center.x,
      y: rotated.y + center.y,
    }
  })
}

function smoothRadialEnvelope(points: Point[], iterations: number) {
  let current = points
  for (let step = 0; step < iterations; step += 1) {
    current = current.map((point, index) => {
      const prev = current[(index - 1 + current.length) % current.length]
      const next = current[(index + 1) % current.length]
      return {
        x: (prev.x + point.x * 3 + next.x) / 5,
        y: (prev.y + point.y * 3 + next.y) / 5,
      }
    })
  }
  return current
}

function fillRadialGaps(radial: number[], fallback: number) {
  let previous = fallback

  for (let index = 0; index < radial.length; index += 1) {
    if (Number.isFinite(radial[index]) && radial[index] > 0) {
      previous = radial[index]
    } else {
      radial[index] = previous
    }
  }

  previous = radial[radial.length - 1] || fallback
  for (let index = radial.length - 1; index >= 0; index -= 1) {
    if (Number.isFinite(radial[index]) && radial[index] > 0) {
      previous = radial[index]
    } else {
      radial[index] = previous
    }
  }

  return radial
}

function buildOuterEnvelope(frames: Point[][]) {
  const bins = 1080
  const radial = Array.from({ length: bins }, () => 0)

  frames.forEach((gear) => {
    gear.forEach((point) => {
      const angle = (Math.atan2(point.y, point.x) + TAU) % TAU
      const radius = Math.hypot(point.x, point.y)
      const bin = Math.floor((angle / TAU) * bins) % bins
      radial[bin] = Math.max(radial[bin], radius)
    })
  })

  const completed = fillRadialGaps(radial, 0)
  const envelope = completed.map((radius, index) =>
    polarToCartesian((index / completed.length) * TAU, radius),
  )

  return smoothRadialEnvelope(envelope, 3)
}

function createRingShell(innerBoundary: Point[], shellThickness: number) {
  return innerBoundary.map((point) => {
    const normal = normalize(point)
    return {
      x: point.x + normal.x * shellThickness,
      y: point.y + normal.y * shellThickness,
    }
  })
}

function formatPoint(point: Point) {
  return `${point.x.toFixed(1)}, ${point.y.toFixed(1)}`
}

function averagePerimeterRadius(points: Point[]) {
  return computeArcLengths(points).total / TAU
}

function buildPlanetFrames(
  planetPitchCurve: Point[],
  planetOutline: Point[],
  carrierRadius: number,
  planetCount: number,
  carrierAngle: number,
  planetSpinAngle: number,
) {
  const pitchFrames: Point[][] = []
  const outlineFrames: Point[][] = []

  for (let planetIndex = 0; planetIndex < planetCount; planetIndex += 1) {
    const orbitAngle = carrierAngle + (planetIndex / planetCount) * TAU
    const center = polarToCartesian(orbitAngle, carrierRadius)
    const rotation = orbitAngle + planetSpinAngle
    pitchFrames.push(transformGear(planetPitchCurve, center, rotation))
    outlineFrames.push(transformGear(planetOutline, center, rotation))
  }

  return { pitchFrames, outlineFrames }
}

function App() {
  const [controlRadii, setControlRadii] = useState(DEFAULT_RADII)
  const [baseBias, setBaseBias] = useState(8)
  const [smoothing, setSmoothing] = useState(20)
  const [toothCount, setToothCount] = useState(28)
  const [toothDepth, setToothDepth] = useState(16)
  const [toothSharpness, setToothSharpness] = useState(22)
  const [planetCount, setPlanetCount] = useState(3)
  const [orbitRadius, setOrbitRadius] = useState(196)
  const [spinRatio, setSpinRatio] = useState(2.55)
  const [ringThickness, setRingThickness] = useState(92)
  const [turningInset, setTurningInset] = useState(18)
  const [selectedTurningIndex, setSelectedTurningIndex] = useState(0)
  const [showPitch, setShowPitch] = useState(true)
  const [showCenters, setShowCenters] = useState(true)
  const [showTracks, setShowTracks] = useState(true)
  const [animationSpeed, setAnimationSpeed] = useState(0.18)
  const [progress, setProgress] = useState(0)
  const [dragIndex, setDragIndex] = useState<number | null>(null)

  const editorRef = useRef<SVGSVGElement | null>(null)

  const pitchCurve = createPitchCurve(controlRadii, baseBias, smoothing)
  const basePitchLength = computeArcLengths(pitchCurve).total
  const toothPitch = basePitchLength / toothCount
  const samplesPerTooth = 40
  const planetPitchCurve = resampleClosedPolyline(pitchCurve, toothCount * samplesPerTooth)
  const planetOutline = createToothedOutline(
    planetPitchCurve,
    toothCount,
    toothDepth,
    toothSharpness,
  )
  const turningCandidates = findTurningCandidates(planetPitchCurve, turningInset)
  const safeTurningIndex =
    turningCandidates.length === 0
      ? 0
      : clamp(selectedTurningIndex, 0, turningCandidates.length - 1)
  const selectedTurningCenter =
    turningCandidates[safeTurningIndex] ?? turningCandidates[0] ?? { x: 0, y: 0, score: 0 }
  const sunScale = 0.33
  const sunBaseCurve = planetPitchCurve.map((point) => scalePoint(point, sunScale))
  const sunPerimeter = computeArcLengths(sunBaseCurve).total
  const sunToothCount = Math.max(8, Math.round(sunPerimeter / toothPitch))
  const sunPitchCurve = resampleClosedPolyline(sunBaseCurve, sunToothCount * samplesPerTooth)
  const sunOutline = createToothedOutline(
    sunPitchCurve,
    sunToothCount,
    toothDepth,
    toothSharpness,
  )
  const averageSunRadius = averagePerimeterRadius(sunPitchCurve)
  const averagePlanetRadius = averagePerimeterRadius(planetPitchCurve)
  const ringToothCount = sunToothCount + toothCount * 2
  const carrierRadius =
    averageSunRadius +
    averagePlanetRadius +
    toothDepth * 0.84 +
    (orbitRadius - 196) * 0.18
  const carrierAngle = progress * TAU
  const sunRotation = carrierAngle * ((sunToothCount + ringToothCount) / sunToothCount)
  const planetSpinRelativeToCarrier =
    -(sunToothCount / toothCount) * (sunRotation - carrierAngle)
  const currentFrames = buildPlanetFrames(
    planetPitchCurve,
    planetOutline,
    carrierRadius,
    planetCount,
    carrierAngle,
    planetSpinRelativeToCarrier,
  )
  const ringPitchBase = buildOuterEnvelope(
    Array.from({ length: 180 }, (_, step) => {
      const sweepCarrier = (step / 180) * TAU
      const sweepSunRotation = sweepCarrier * ((sunToothCount + ringToothCount) / sunToothCount)
      const sweepSpin = -(sunToothCount / toothCount) * (sweepSunRotation - sweepCarrier)
      return buildPlanetFrames(
        planetPitchCurve,
        planetOutline,
        carrierRadius,
        planetCount,
        sweepCarrier,
        sweepSpin,
      ).pitchFrames
    }).flat(),
  )
  const ringPitchCurve = resampleClosedPolyline(ringPitchBase, ringToothCount * samplesPerTooth)
  const ringInnerBoundary = createToothedOutline(
    ringPitchCurve,
    ringToothCount,
    toothDepth,
    toothSharpness,
    true,
  )
  const ringOuterBoundary = createRingShell(ringInnerBoundary, ringThickness)
  const rotatedSunOutline = sunOutline.map((point) => rotatePoint(point, sunRotation))
  const rotatedSunPitch = sunPitchCurve.map((point) => rotatePoint(point, sunRotation))

  useEffect(() => {
    let frame = 0
    let last = performance.now()

    const animate = (now: number) => {
      const delta = (now - last) / 1000
      last = now
      setProgress((value) => (value + delta * animationSpeed) % 1)
      frame = window.requestAnimationFrame(animate)
    }

    frame = window.requestAnimationFrame(animate)
    return () => window.cancelAnimationFrame(frame)
  }, [animationSpeed])

  useEffect(() => {
    if (dragIndex === null) {
      return
    }

    const handlePointerMove = (event: PointerEvent) => {
      if (!editorRef.current) {
        return
      }

      const bounds = editorRef.current.getBoundingClientRect()
      const x = ((event.clientX - bounds.left) / bounds.width) * EDITOR_SIZE - EDITOR_CENTER
      const y = ((event.clientY - bounds.top) / bounds.height) * EDITOR_SIZE - EDITOR_CENTER
      const radius = clamp(Math.hypot(x, y), 68, 228)

      setControlRadii((current) =>
        current.map((value, index) => (index === dragIndex ? radius : value)),
      )
    }

    const handlePointerUp = () => {
      setDragIndex(null)
    }

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', handlePointerUp)
    return () => {
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', handlePointerUp)
    }
  }, [dragIndex])

  const exportSvg = () => {
    const paths = []
    const ringOuterPath = pointsToSmoothPath(ringOuterBoundary, CENTER, CENTER)
    const ringInnerPath = pointsToSmoothPath([...ringInnerBoundary].reverse(), CENTER, CENTER)
    const trackRadius = carrierRadius
    const pitchPath = pointsToSmoothPath(planetPitchCurve, CENTER, CENTER)

    paths.push(
      `<path d="${ringOuterPath} ${ringInnerPath}" fill="#e9e0d1" fill-rule="evenodd" stroke="#201b17" stroke-width="2" />`,
    )
    paths.push(
      `<path d="${pointsToSmoothPath(rotatedSunOutline, CENTER, CENTER)}" fill="#f0b758" stroke="#f7f0df" stroke-width="1.5" />`,
    )

    for (const transformed of currentFrames.outlineFrames) {
      paths.push(
        `<path d="${pointsToSmoothPath(transformed, CENTER, CENTER)}" fill="#0d1318" stroke="#f7f0df" stroke-width="1.5" />`,
      )
    }

    paths.push(`<path d="${pitchPath}" fill="none" stroke="#ca8f32" stroke-width="1.5" opacity="0.7" />`)
    paths.push(
      `<circle cx="${CENTER}" cy="${CENTER}" r="${trackRadius}" fill="none" stroke="#786a58" stroke-width="1.25" stroke-dasharray="6 8" opacity="0.55" />`,
    )

    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${VIEWBOX_SIZE} ${VIEWBOX_SIZE}"><rect width="100%" height="100%" fill="#090d11" />${paths.join('')}</svg>`
    const blob = new Blob([svg], { type: 'image/svg+xml' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'astronomer-non-circular-planetary.svg'
    link.click()
    URL.revokeObjectURL(url)
  }

  const orbitTrack = currentFrames.outlineFrames

  return (
    <div className="app-shell">
      <aside className="control-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Astronomer Prototype</p>
            <h1>Non-circular planetary gear designer</h1>
          </div>
          <button className="export-button" onClick={exportSvg}>
            Export SVG
          </button>
        </div>

        <section className="panel-card">
          <div className="card-title-row">
            <h2>Planet design</h2>
            <span>{controlRadii.length} control points</span>
          </div>
          <svg
            className="editor"
            ref={editorRef}
            viewBox={`0 0 ${EDITOR_SIZE} ${EDITOR_SIZE}`}
            role="img"
            aria-label="Planet control-point editor"
          >
            <defs>
              <radialGradient id="editor-glow">
                <stop offset="0%" stopColor="#172730" />
                <stop offset="100%" stopColor="#0c1116" />
              </radialGradient>
            </defs>
            <rect width={EDITOR_SIZE} height={EDITOR_SIZE} rx="24" fill="url(#editor-glow)" />
            <circle
              cx={EDITOR_CENTER}
              cy={EDITOR_CENTER}
              r={92}
              fill="none"
              stroke="#35515f"
              strokeDasharray="6 10"
            />
            <circle
              cx={EDITOR_CENTER}
              cy={EDITOR_CENTER}
              r={150}
              fill="none"
              stroke="#22323c"
              strokeDasharray="3 7"
            />
            <path
              d={pointsToSmoothPath(planetPitchCurve, EDITOR_CENTER, EDITOR_CENTER)}
              className="pitch-curve"
            />
            <path
              d={pointsToSmoothPath(planetOutline, EDITOR_CENTER, EDITOR_CENTER)}
              className="planet-outline"
            />
            {turningCandidates.map((candidate, index) => (
              <circle
                key={`${candidate.x}-${candidate.y}`}
                cx={candidate.x + EDITOR_CENTER}
                cy={candidate.y + EDITOR_CENTER}
                r={index === safeTurningIndex ? 7 : 4}
                className={index === safeTurningIndex ? 'turning-candidate selected' : 'turning-candidate'}
                onClick={() => setSelectedTurningIndex(index)}
              />
            ))}
            {controlRadii.map((radius, index) => {
              const angle = (index / controlRadii.length) * TAU
              const harmonic = Math.cos(angle * 3) * baseBias + Math.sin(angle * 2) * baseBias * 0.45
              const point = polarToCartesian(angle, radius + harmonic)

              return (
                <g key={`handle-${index}`}>
                  <line
                    x1={EDITOR_CENTER}
                    y1={EDITOR_CENTER}
                    x2={point.x + EDITOR_CENTER}
                    y2={point.y + EDITOR_CENTER}
                    className="handle-spoke"
                  />
                  <circle
                    cx={point.x + EDITOR_CENTER}
                    cy={point.y + EDITOR_CENTER}
                    r={10}
                    className="control-handle"
                    onPointerDown={() => setDragIndex(index)}
                  />
                </g>
              )
            })}
          </svg>
          <p className="card-note">
            Drag the handles to shape the planet. Green points show feasible internal turning centers
            extracted from the planet interior.
          </p>
        </section>

        <section className="panel-card grid-card">
          <div className="card-title-row">
            <h2>Geometry</h2>
            <span>Pitch and tooth controls</span>
          </div>
          <label>
            Base bias
            <input type="range" min="0" max="24" value={baseBias} onChange={(event) => setBaseBias(Number(event.target.value))} />
            <strong>{baseBias}</strong>
          </label>
          <label>
            Smoothing
            <input
              type="range"
              min="4"
              max="48"
              value={smoothing}
              onChange={(event) => setSmoothing(Number(event.target.value))}
            />
            <strong>{smoothing}</strong>
          </label>
          <label>
            Tooth count
            <input
              type="range"
              min="12"
              max="42"
              value={toothCount}
              onChange={(event) => setToothCount(Number(event.target.value))}
            />
            <strong>{toothCount}</strong>
          </label>
          <label>
            Tooth depth
            <input
              type="range"
              min="4"
              max="28"
              value={toothDepth}
              onChange={(event) => setToothDepth(Number(event.target.value))}
            />
            <strong>{toothDepth}</strong>
          </label>
          <label>
            Tooth sharpness
            <input
              type="range"
              min="6"
              max="38"
              value={toothSharpness}
              onChange={(event) => setToothSharpness(Number(event.target.value))}
            />
            <strong>{toothSharpness}</strong>
          </label>
          <label>
            Turning inset
            <input
              type="range"
              min="8"
              max="36"
              value={turningInset}
              onChange={(event) => setTurningInset(Number(event.target.value))}
            />
            <strong>{turningInset}</strong>
          </label>
        </section>

        <section className="panel-card grid-card">
          <div className="card-title-row">
            <h2>Mechanism</h2>
            <span>Ring generation</span>
          </div>
          <label>
            Planets
            <input
              type="range"
              min="2"
              max="5"
              value={planetCount}
              onChange={(event) => setPlanetCount(Number(event.target.value))}
            />
            <strong>{planetCount}</strong>
          </label>
          <label>
            Orbit radius
            <input
              type="range"
              min="120"
              max="240"
              value={orbitRadius}
              onChange={(event) => setOrbitRadius(Number(event.target.value))}
            />
            <strong>{orbitRadius}</strong>
          </label>
          <label>
            Spin ratio
            <input
              type="range"
              min="1.2"
              max="4.8"
              step="0.01"
              value={spinRatio}
              onChange={(event) => setSpinRatio(Number(event.target.value))}
            />
            <strong>{spinRatio.toFixed(2)}x</strong>
          </label>
          <label>
            Ring thickness
            <input
              type="range"
              min="56"
              max="132"
              value={ringThickness}
              onChange={(event) => setRingThickness(Number(event.target.value))}
            />
            <strong>{ringThickness}</strong>
          </label>
          <label>
            Animation speed
            <input
              type="range"
              min="0.04"
              max="0.55"
              step="0.01"
              value={animationSpeed}
              onChange={(event) => setAnimationSpeed(Number(event.target.value))}
            />
            <strong>{animationSpeed.toFixed(2)} rps</strong>
          </label>
          <div className="toggle-row">
            <label className="checkbox">
              <input type="checkbox" checked={showPitch} onChange={() => setShowPitch((value) => !value)} />
              Pitch curve
            </label>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={showCenters}
                onChange={() => setShowCenters((value) => !value)}
              />
              Turning centers
            </label>
            <label className="checkbox">
              <input type="checkbox" checked={showTracks} onChange={() => setShowTracks((value) => !value)} />
              Orbit track
            </label>
          </div>
        </section>

        <section className="panel-card metrics-card">
          <div>
            <span>Selected center</span>
            <strong>{formatPoint(selectedTurningCenter)}</strong>
          </div>
          <div>
            <span>Clearance score</span>
            <strong>{selectedTurningCenter.score?.toFixed(1) ?? '0.0'}</strong>
          </div>
          <div>
            <span>Teeth P/S/R</span>
            <strong>{`${toothCount} / ${sunToothCount} / ${ringToothCount}`}</strong>
          </div>
        </section>
      </aside>

      <main className="viewport-panel">
        <div className="viewport-header">
          <div>
            <p className="eyebrow">Live mechanism</p>
            <h2>Animated ring / planet assembly</h2>
          </div>
          <p className="status-copy">
            The outer ring is generated as the swept envelope of the rotating planets around the
            fixed sun gear, then re-toothed on the same shared pitch.
          </p>
        </div>

        <svg
          className="viewport"
          viewBox={`0 0 ${VIEWBOX_SIZE} ${VIEWBOX_SIZE}`}
          role="img"
          aria-label="Generated non-circular planetary gear mechanism"
        >
          <defs>
            <radialGradient id="viewport-bg" cx="50%" cy="42%">
              <stop offset="0%" stopColor="#17242d" />
              <stop offset="100%" stopColor="#081015" />
            </radialGradient>
            <linearGradient id="ring-fill" x1="0%" x2="100%" y1="0%" y2="100%">
              <stop offset="0%" stopColor="#efe4d3" />
              <stop offset="100%" stopColor="#b28d63" />
            </linearGradient>
            <linearGradient id="planet-fill" x1="0%" x2="100%" y1="0%" y2="100%">
              <stop offset="0%" stopColor="#090d11" />
              <stop offset="100%" stopColor="#202830" />
            </linearGradient>
          </defs>
          <rect width={VIEWBOX_SIZE} height={VIEWBOX_SIZE} rx="34" fill="url(#viewport-bg)" />

          <path
            d={`${pointsToSmoothPath(ringOuterBoundary, CENTER, CENTER)} ${pointsToSmoothPath(
              [...ringInnerBoundary].reverse(),
              CENTER,
              CENTER,
            )}`}
            fill="url(#ring-fill)"
            fillRule="evenodd"
            stroke="#f7f2e8"
            strokeWidth="2.5"
          />

          {showTracks ? (
            <circle
              cx={CENTER}
              cy={CENTER}
              r={carrierRadius}
              fill="none"
              stroke="#5b6d76"
              strokeDasharray="9 12"
              opacity="0.55"
            />
          ) : null}

          {showPitch ? (
            <>
              <path
                d={pointsToSmoothPath(planetPitchCurve, CENTER, CENTER)}
                fill="none"
                stroke="#f2b14a"
                strokeWidth="1.5"
                opacity="0.42"
              />
              <path
                d={pointsToSmoothPath(ringPitchCurve, CENTER, CENTER)}
                fill="none"
                stroke="#8fd3ff"
                strokeWidth="1.4"
                opacity="0.34"
              />
              <path
                d={pointsToSmoothPath(rotatedSunPitch, CENTER, CENTER)}
                fill="none"
                stroke="#ffe082"
                strokeWidth="1.3"
                opacity="0.45"
              />
            </>
          ) : null}

          <path
            d={pointsToSmoothPath(rotatedSunOutline, CENTER, CENTER)}
            fill="#f0b758"
            stroke="#fff2d8"
            strokeWidth="1.5"
          />

          {orbitTrack.map((planet, index) => (
            <path
              key={`planet-${index}`}
              d={pointsToSmoothPath(translatePath(planet, { x: CENTER, y: CENTER }), 0, 0)}
              fill="url(#planet-fill)"
              stroke="#f0eadf"
              strokeWidth="1.3"
            />
          ))}

          {showCenters
            ? turningCandidates.map((candidate, index) => (
                <circle
                  key={`center-${candidate.x}-${candidate.y}`}
                  cx={candidate.x + CENTER}
                  cy={candidate.y + CENTER}
                  r={index === safeTurningIndex ? 7 : 4}
                  fill={index === safeTurningIndex ? '#5af2b3' : '#2c7f62'}
                  opacity={0.85}
                />
              ))
            : null}

          <circle cx={CENTER} cy={CENTER} r="8" fill="#d8a455" />
          <text x="28" y="42" className="viewport-label">{`Pitch: ${toothPitch.toFixed(2)}`}</text>
          <text
            x="28"
            y="68"
            className="viewport-label"
          >{`Avg radii P/S: ${averagePlanetRadius.toFixed(1)} / ${averageSunRadius.toFixed(1)}`}</text>
        </svg>
      </main>
    </div>
  )
}

export default App
