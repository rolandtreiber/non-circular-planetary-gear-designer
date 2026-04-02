import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

type Point = {
  x: number
  y: number
}

type TurningCandidate = Point & {
  score: number
}

type SolvabilityGraph = {
  path: Point[]
  samples: Array<{
    point: Point
    v: number
    externalDistance: number
    internalDistance: number
    difference: number
  }>
  intersections: TurningCandidate[]
}

const TAU = Math.PI * 2
const VIEWBOX_SIZE = 880
const EDITOR_SIZE = 320
const CENTER = VIEWBOX_SIZE / 2
const EDITOR_CENTER = EDITOR_SIZE / 2
const GEARIFY_SAMPLE_RADII = [138, 148, 172, 148, 138, 148, 172, 148, 138, 148, 172, 148]
const DEFAULT_BASE_BIAS = 0
const DEFAULT_SMOOTHING = 30
const DEFAULT_TOOTH_COUNT = 24
const DEFAULT_TOOTH_DEPTH = 12
const DEFAULT_TOOTH_SHARPNESS = 18
const DEFAULT_ORBIT_RADIUS = 172
const DEFAULT_RING_THICKNESS = 88
const DEFAULT_TURNING_INSET = 16

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

function computeCentroid(points: Point[]) {
  const sum = points.reduce(
    (acc, point) => ({ x: acc.x + point.x, y: acc.y + point.y }),
    { x: 0, y: 0 },
  )
  return {
    x: sum.x / points.length,
    y: sum.y / points.length,
  }
}

function computePrincipalAxis(points: Point[]) {
  const centroid = computeCentroid(points)
  let xx = 0
  let xy = 0
  let yy = 0

  points.forEach((point) => {
    const dx = point.x - centroid.x
    const dy = point.y - centroid.y
    xx += dx * dx
    xy += dx * dy
    yy += dy * dy
  })

  const trace = xx + yy
  const det = xx * yy - xy * xy
  const eigen = trace / 2 + Math.sqrt(Math.max(0, (trace * trace) / 4 - det))
  const axis =
    Math.abs(xy) > 1e-6
      ? normalize({ x: eigen - yy, y: xy })
      : xx >= yy
        ? { x: 1, y: 0 }
        : { x: 0, y: 1 }

  return { centroid, axis, normal: { x: -axis.y, y: axis.x } }
}

function projectOnAxis(point: Point, axis: Point) {
  return point.x * axis.x + point.y * axis.y
}

function buildSolvabilityPath(points: Point[], minInset: number) {
  const { centroid, axis, normal } = computePrincipalAxis(points)
  const projections = points.map((point) => projectOnAxis(point, axis))
  const minT = Math.min(...projections)
  const maxT = Math.max(...projections)
  const samples: Point[] = []
  const scanCount = 80

  for (let index = 0; index < scanCount; index += 1) {
    const t = minT + ((maxT - minT) * index) / (scanCount - 1)
    const intersections: number[] = []

    for (let pointIndex = 0; pointIndex < points.length; pointIndex += 1) {
      const start = points[pointIndex]
      const end = points[(pointIndex + 1) % points.length]
      const startT = projectOnAxis(start, axis)
      const endT = projectOnAxis(end, axis)
      const crosses =
        (startT <= t && endT > t) || (endT <= t && startT > t) || Math.abs(startT - t) < 1e-6

      if (!crosses) {
        continue
      }

      const deltaT = endT - startT
      const lerp = Math.abs(deltaT) < 1e-6 ? 0 : (t - startT) / deltaT
      const point = {
        x: start.x + (end.x - start.x) * lerp,
        y: start.y + (end.y - start.y) * lerp,
      }
      intersections.push(projectOnAxis(point, normal))
    }

    if (intersections.length < 2) {
      continue
    }

    intersections.sort((a, b) => a - b)
    let bestMidpoint: Point | null = null
    let bestSpan = 0

    for (let spanIndex = 0; spanIndex + 1 < intersections.length; spanIndex += 2) {
      const startS = intersections[spanIndex]
      const endS = intersections[spanIndex + 1]
      const span = endS - startS
      if (span <= bestSpan) {
        continue
      }
      const midS = (startS + endS) / 2
      const candidate = {
        x: axis.x * t + normal.x * midS,
        y: axis.y * t + normal.y * midS,
      }
      let edgeDistance = Infinity
      for (let edgeIndex = 0; edgeIndex < points.length; edgeIndex += 1) {
        edgeDistance = Math.min(
          edgeDistance,
          distancePointToSegment(candidate, points[edgeIndex], points[(edgeIndex + 1) % points.length]),
        )
      }
      if (edgeDistance >= minInset) {
        bestSpan = span
        bestMidpoint = candidate
      }
    }

    if (bestMidpoint) {
      samples.push({
        x: bestMidpoint.x + centroid.x * 0,
        y: bestMidpoint.y + centroid.y * 0,
      })
    }
  }

  return samples.length >= 4 ? smoothRadialEnvelope(samples, 1) : samples
}

function findSolvableTurningCandidates(
  outline: Point[],
  toothPitch: number,
  planetToothCount: number,
  minInset: number,
) {
  const path = buildSolvabilityPath(outline, minInset)
  if (path.length < 2) {
    return { path: [], samples: [], intersections: [] } satisfies SolvabilityGraph
  }

  const sunToothCount = planetToothCount * 2
  const ringToothCount = sunToothCount * 2
  const targetSunRadius = (toothPitch * sunToothCount) / TAU
  const targetRingRadius = (toothPitch * ringToothCount) / TAU

  const samples = path.map((point, index) => {
    const relativeOutline = outline.map((outlinePoint) => ({
      x: outlinePoint.x - point.x,
      y: outlinePoint.y - point.y,
    }))
    const outerOffset = averagePerimeterRadius(relativeOutline)
    const innerProfile = extractTurningProfile(outline, point, 'inner').map((profilePoint) => ({
      x: profilePoint.x - point.x,
      y: profilePoint.y - point.y,
    }))
    const innerOffset = averagePerimeterRadius(innerProfile)
    const externalDistance = targetRingRadius - outerOffset
    const internalDistance = targetSunRadius + innerOffset

    return {
      point,
      v: path.length === 1 ? 0 : index / (path.length - 1),
      externalDistance,
      internalDistance,
      difference: externalDistance - internalDistance,
    }
  })

  const intersections: TurningCandidate[] = []

  for (let index = 0; index + 1 < samples.length; index += 1) {
    const current = samples[index]
    const next = samples[index + 1]
    const currentDifference = current.difference
    const nextDifference = next.difference

    if (Math.abs(currentDifference) < 1e-3) {
      intersections.push({ ...current.point, score: 1 / (1 + Math.abs(currentDifference)) })
      continue
    }

    if (currentDifference * nextDifference > 0) {
      continue
    }

    const blend = Math.abs(currentDifference) / (Math.abs(currentDifference) + Math.abs(nextDifference) || 1)
    intersections.push({
      x: current.point.x + (next.point.x - current.point.x) * blend,
      y: current.point.y + (next.point.y - current.point.y) * blend,
      score: 1 / (1 + Math.abs(currentDifference) + Math.abs(nextDifference)),
    })
  }

  return { path, samples, intersections } satisfies SolvabilityGraph
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
    if (Number.isFinite(radial[index])) {
      previous = radial[index]
    } else {
      radial[index] = previous
    }
  }

  previous = radial[radial.length - 1] || fallback
  for (let index = radial.length - 1; index >= 0; index -= 1) {
    if (Number.isFinite(radial[index])) {
      previous = radial[index]
    } else {
      radial[index] = previous
    }
  }

  return radial
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

function scalePoints(points: Point[], factor: number) {
  return points.map((point) => ({
    x: point.x * factor,
    y: point.y * factor,
  }))
}

function scaleCurveToLength(points: Point[], targetLength: number) {
  const currentLength = computeArcLengths(points).total || 1
  return scalePoints(points, targetLength / currentLength)
}

function createCirclePoints(radius: number, count: number) {
  return Array.from({ length: count }, (_, index) =>
    polarToCartesian((index / count) * TAU, radius),
  )
}

function findRepeatTurns(values: number[], maxTurns = 120, tolerance = 1e-6) {
  for (let turns = 1; turns <= maxTurns; turns += 1) {
    const repeats = values.every((value) => Math.abs(value * turns - Math.round(value * turns)) < tolerance)
    if (repeats) {
      return turns
    }
  }

  return 1
}

function extractTurningProfile(points: Point[], turningCenter: Point, mode: 'inner' | 'outer') {
  const bins = 720
  const radial = Array.from({ length: bins }, () =>
    mode === 'outer' ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY,
  )

  points.forEach((point) => {
    const relative = {
      x: point.x - turningCenter.x,
      y: point.y - turningCenter.y,
    }
    const angle = normalizeAngle(Math.atan2(relative.y, relative.x))
    const radius = Math.hypot(relative.x, relative.y)
    const bin = Math.floor((angle / TAU) * bins) % bins
    radial[bin] =
      mode === 'outer'
        ? Math.max(radial[bin], radius)
        : Math.min(radial[bin], radius)
  })

  const fallback =
    mode === 'outer'
      ? Math.max(...radial.filter((value) => Number.isFinite(value)), 1)
      : Math.min(...radial.filter((value) => Number.isFinite(value)), 1)
  const completed = fillRadialGaps(
    radial.map((value) => (Number.isFinite(value) ? value : fallback)),
    fallback,
  )

  return smoothRadialEnvelope(
    completed.map((radius, index) => ({
      x: turningCenter.x + Math.cos((index / bins) * TAU) * radius,
      y: turningCenter.y + Math.sin((index / bins) * TAU) * radius,
    })),
    mode === 'outer' ? 2 : 4,
  )
}

function formatPoint(point: Point) {
  return `${point.x.toFixed(1)}, ${point.y.toFixed(1)}`
}

function averagePerimeterRadius(points: Point[]) {
  return computeArcLengths(points).total / TAU
}

function averageEdgeLength(points: Point[]) {
  return computeArcLengths(points).total / Math.max(points.length, 1)
}

function maxEdgeLength(points: Point[]) {
  let maxLength = 0
  for (let index = 0; index < points.length; index += 1) {
    maxLength = Math.max(maxLength, distance(points[index], points[(index + 1) % points.length]))
  }
  return maxLength
}

function maxRadialJump(points: Point[]) {
  let maxJump = 0
  for (let index = 0; index < points.length; index += 1) {
    const currentRadius = Math.hypot(points[index].x, points[index].y)
    const nextRadius = Math.hypot(points[(index + 1) % points.length].x, points[(index + 1) % points.length].y)
    maxJump = Math.max(maxJump, Math.abs(nextRadius - currentRadius))
  }
  return maxJump
}

function normalizeAngle(angle: number) {
  return ((angle % TAU) + TAU) % TAU
}

function buildConjugateEnvelope(
  frames: Point[][],
  mode: 'inner' | 'outer',
  fallback: number,
  bins = 1440,
  smoothingIterations = 2,
) {
  const radial = Array.from({ length: bins }, () =>
    mode === 'outer' ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY,
  )

  frames.forEach((frame) => {
    frame.forEach((point) => {
      const angle = normalizeAngle(Math.atan2(point.y, point.x))
      const radius = Math.hypot(point.x, point.y)
      const bin = Math.floor((angle / TAU) * bins) % bins
      radial[bin] =
        mode === 'outer'
          ? Math.max(radial[bin], radius)
          : Math.min(radial[bin], radius)
    })
  })

  const completed = fillRadialGaps(
    radial.map((value) => (Number.isFinite(value) ? value : fallback)),
    fallback,
  )
  const curve = completed.map((radius, index) =>
    polarToCartesian((index / completed.length) * TAU, radius),
  )

  return smoothingIterations > 0 ? smoothRadialEnvelope(curve, smoothingIterations) : curve
}

function buildPlanetFrames(
  planetPitchCurve: Point[],
  planetOutline: Point[],
  turningCenter: Point,
  carrierRadius: number,
  planetCount: number,
  carrierAngle: number,
  planetSpinFactor: number,
) {
  const pitchFrames: Point[][] = []
  const outlineFrames: Point[][] = []

  for (let planetIndex = 0; planetIndex < planetCount; planetIndex += 1) {
    const orbitAngle = carrierAngle + (planetIndex / planetCount) * TAU
    const center = polarToCartesian(orbitAngle, carrierRadius)
    const rotation = orbitAngle * planetSpinFactor
    const transformedPitch = planetPitchCurve.map((point) => {
      const rotated = rotatePoint(
        { x: point.x - turningCenter.x, y: point.y - turningCenter.y },
        rotation,
      )
      return {
        x: center.x + rotated.x,
        y: center.y + rotated.y,
      }
    })
    const transformedOutline = planetOutline.map((point) => {
      const rotated = rotatePoint(
        { x: point.x - turningCenter.x, y: point.y - turningCenter.y },
        rotation,
      )
      return {
        x: center.x + rotated.x,
        y: center.y + rotated.y,
      }
    })
    pitchFrames.push(transformedPitch)
    outlineFrames.push(transformedOutline)
  }

  return { pitchFrames, outlineFrames }
}

function samplePlanetMotion(
  planetOutline: Point[],
  turningCenter: Point,
  carrierRadius: number,
  planetSpinFactor: number,
  sampleCount: number,
) {
  return Array.from({ length: sampleCount }, (_, step) => {
    const carrierAngle = (step / sampleCount) * TAU
    const center = polarToCartesian(carrierAngle, carrierRadius)
    const rotation = carrierAngle * planetSpinFactor

    return planetOutline.map((point) => {
      const rotated = rotatePoint(
        { x: point.x - turningCenter.x, y: point.y - turningCenter.y },
        rotation,
      )
      return {
        x: center.x + rotated.x,
        y: center.y + rotated.y,
      }
    })
  })
}

type MechanismGeometry = {
  carrierRadius: number
  planetSpinFactor: number
  sunSpinFactor: number
  ringToothCount: number
  sunToothCount: number
  ringInnerBoundary: Point[]
  ringOuterBoundary: Point[]
  ringPitchCurve: Point[]
  sunOutline: Point[]
  sunPitchCurve: Point[]
  ringPitchError: number
  sunPitchError: number
  sunContinuous: boolean
  solvable: boolean
}

type RingCarveResult = {
  carrierRadius: number
  planetSpinFactor: number
  ringToothCount: number
  ringInnerBoundary: Point[]
  ringOuterBoundary: Point[]
  ringPitchCurve: Point[]
  ringPitchError: number
  baseInnerBlank: Point[]
  baseOuterBlank: Point[]
  targetSunPitchLength: number
  targetSunRadius: number
  solvable: boolean
}

type BuildStage = 'ring' | 'sun' | 'assembly'

function searchRingCarve({
  cutterOutline,
  turningCenter,
  planetToothCount,
  planetCount,
  toothPitch,
  orbitRadius,
  ringThickness,
}: {
  cutterOutline: Point[]
  turningCenter: Point
  planetToothCount: number
  planetCount: number
  toothPitch: number
  orbitRadius: number
  ringThickness: number
}): RingCarveResult {
  const averagePlanetRadius = averagePerimeterRadius(cutterOutline)
  const outerOffsetRadius = averagePerimeterRadius(
    cutterOutline.map((point) => ({ x: point.x - turningCenter.x, y: point.y - turningCenter.y })),
  )
  const sampleCount = 240
  const sunToothCountTarget = planetToothCount * 2
  const ringToothCountTarget = sunToothCountTarget * 2
  const targetSunPitchLength = toothPitch * sunToothCountTarget
  const targetRingPitchLength = toothPitch * ringToothCountTarget
  const targetSunRadius = targetSunPitchLength / TAU
  const targetRingRadius = targetRingPitchLength / TAU
  const targetCarrierRadius = (targetSunRadius + targetRingRadius - outerOffsetRadius) / 2
  const radiusCandidates = Array.from({ length: 21 }, (_, index) =>
    clamp(targetCarrierRadius - 42 + index * 4.2, averagePlanetRadius * 0.9, targetCarrierRadius + 42),
  )
  const planetSpinFactor = -planetCount

  let best: (RingCarveResult & { score: number }) | null = null

  radiusCandidates.forEach((carrierRadius) => {
    const ringFrames = samplePlanetMotion(
      cutterOutline,
      turningCenter,
      carrierRadius,
      planetSpinFactor,
      sampleCount,
    )
    const ringEnvelope = buildConjugateEnvelope(
      ringFrames,
      'outer',
      carrierRadius + averagePlanetRadius,
      2880,
      0,
    )
    const ringPitchCurve = resampleClosedPolyline(
      ringEnvelope,
      Math.max(720, ringToothCountTarget * 48),
    )
    const ringPitchLength = computeArcLengths(ringPitchCurve).total
    const ringPitchError = Math.abs(ringPitchLength / toothPitch - ringToothCountTarget)

    const continuityPenalty = Math.abs(computeSignedArea(ringPitchCurve)) < 1 ? 1000 : 0
    const score =
      ringPitchError * 10 +
      Math.abs(carrierRadius - targetCarrierRadius) * 0.12 +
      Math.abs(averagePerimeterRadius(ringPitchCurve) - targetRingRadius) * 0.02 +
      continuityPenalty

    if (!best || score < best.score) {
      best = {
        carrierRadius,
        planetSpinFactor,
        ringToothCount: ringToothCountTarget,
        ringInnerBoundary: ringPitchCurve,
        ringOuterBoundary: createRingShell(ringPitchCurve, ringThickness),
        ringPitchCurve,
        ringPitchError,
        baseInnerBlank: createCirclePoints(averagePerimeterRadius(ringPitchCurve), 360),
        baseOuterBlank: createCirclePoints(averagePerimeterRadius(ringPitchCurve) + ringThickness, 360),
        targetSunPitchLength,
        targetSunRadius,
        solvable: ringPitchError < 0.14,
        score,
      }
    }
  })

  if (best) {
    return best
  }

  const fallbackRing = createRingShell(cutterOutline, ringThickness * 2.2)
  return {
    carrierRadius: targetCarrierRadius || orbitRadius,
    planetSpinFactor: -2,
    ringToothCount: ringToothCountTarget,
    ringInnerBoundary: fallbackRing,
    ringOuterBoundary: createRingShell(fallbackRing, ringThickness),
    ringPitchCurve: fallbackRing,
    ringPitchError: Infinity,
    baseInnerBlank: createCirclePoints(targetRingRadius, 360),
    baseOuterBlank: createCirclePoints(targetRingRadius + ringThickness, 360),
    targetSunPitchLength,
    targetSunRadius,
    solvable: false,
  }
}

function buildMechanismFromRingCarve({
  ringCarve,
  innerCutter,
  turningCenter,
  planetToothCount,
  planetCount,
  toothPitch,
}: {
  ringCarve: RingCarveResult
  innerCutter: Point[]
  turningCenter: Point
  planetToothCount: number
  planetCount: number
  toothPitch: number
}): MechanismGeometry {
  const sampleCount = 240
  const sunToothCount = planetToothCount * 2
  const averagePlanetRadius = averagePerimeterRadius(innerCutter)
  const sunSpinFactor = planetCount * 2
  const innerFrames = samplePlanetMotion(
    innerCutter,
    turningCenter,
    ringCarve.carrierRadius,
    ringCarve.planetSpinFactor,
    sampleCount,
  )
  const sunFrames = innerFrames.map((frame, step) => {
    const carrierAngle = (step / sampleCount) * TAU
    const sunRotation = carrierAngle * sunSpinFactor
    return frame.map((point) => rotatePoint(point, -sunRotation))
  })
  const sunEnvelope = buildConjugateEnvelope(
    sunFrames,
    'inner',
    Math.max(averagePlanetRadius * 0.32, ringCarve.carrierRadius - averagePlanetRadius),
  )
  const scaledSunEnvelope = scaleCurveToLength(sunEnvelope, ringCarve.targetSunPitchLength)
  const sunPitchCurve = resampleClosedPolyline(
    scaledSunEnvelope,
    Math.max(480, sunToothCount * 48),
  )
  const sunPitchLength = computeArcLengths(sunPitchCurve).total
  const sunPitchError = Math.abs(sunPitchLength / toothPitch - sunToothCount)
  const sunAverageRadius = averagePerimeterRadius(sunPitchCurve)
  const sunContinuous =
    Math.abs(computeSignedArea(sunPitchCurve)) > 1 &&
    maxEdgeLength(sunPitchCurve) / Math.max(averageEdgeLength(sunPitchCurve), 1e-6) < 3 &&
    maxRadialJump(sunPitchCurve) / Math.max(sunAverageRadius, 1e-6) < 0.22 &&
    Math.abs(sunAverageRadius - ringCarve.targetSunRadius) / Math.max(ringCarve.targetSunRadius, 1e-6) < 0.18

  return {
    carrierRadius: ringCarve.carrierRadius,
    planetSpinFactor: ringCarve.planetSpinFactor,
    sunSpinFactor,
    ringToothCount: ringCarve.ringToothCount,
    sunToothCount,
    ringInnerBoundary: ringCarve.ringInnerBoundary,
    ringOuterBoundary: ringCarve.ringOuterBoundary,
    ringPitchCurve: ringCarve.ringPitchCurve,
    sunOutline: sunPitchCurve,
    sunPitchCurve,
    ringPitchError: ringCarve.ringPitchError,
    sunPitchError,
    sunContinuous,
    solvable: ringCarve.solvable && sunPitchError < 0.18 && sunContinuous,
  }
}

function App() {
  const [controlRadii, setControlRadii] = useState(GEARIFY_SAMPLE_RADII)
  const [baseBias, setBaseBias] = useState(DEFAULT_BASE_BIAS)
  const [smoothing, setSmoothing] = useState(DEFAULT_SMOOTHING)
  const [toothCount, setToothCount] = useState(DEFAULT_TOOTH_COUNT)
  const [toothDepth, setToothDepth] = useState(DEFAULT_TOOTH_DEPTH)
  const [toothSharpness, setToothSharpness] = useState(DEFAULT_TOOTH_SHARPNESS)
  const [planetCount, setPlanetCount] = useState(3)
  const [orbitRadius, setOrbitRadius] = useState(DEFAULT_ORBIT_RADIUS)
  const [ringThickness, setRingThickness] = useState(DEFAULT_RING_THICKNESS)
  const [turningInset, setTurningInset] = useState(DEFAULT_TURNING_INSET)
  const [selectedTurningIndex, setSelectedTurningIndex] = useState(0)
  const [buildStage, setBuildStage] = useState<BuildStage>('ring')
  const [showPitch, setShowPitch] = useState(true)
  const [showCenters, setShowCenters] = useState(true)
  const [showTracks, setShowTracks] = useState(true)
  const [animationSpeed, setAnimationSpeed] = useState(0.18)
  const [progress, setProgress] = useState(0)
  const [dragIndex, setDragIndex] = useState<number | null>(null)

  const editorRef = useRef<SVGSVGElement | null>(null)

  const samplesPerTooth = 40
  const pitchCurve = useMemo(
    () => createPitchCurve(controlRadii, baseBias, smoothing),
    [controlRadii, baseBias, smoothing],
  )
  const basePitchLength = useMemo(() => computeArcLengths(pitchCurve).total, [pitchCurve])
  const toothPitch = basePitchLength / toothCount
  const planetPitchCurve = useMemo(
    () => resampleClosedPolyline(pitchCurve, toothCount * samplesPerTooth),
    [pitchCurve, toothCount],
  )
  const planetOutline = useMemo(
    () =>
      createToothedOutline(planetPitchCurve, toothCount, toothDepth, toothSharpness),
    [planetPitchCurve, toothCount, toothDepth, toothSharpness],
  )
  const solvabilityGraph = useMemo(
    () => findSolvableTurningCandidates(planetOutline, toothPitch, toothCount, turningInset),
    [planetOutline, toothPitch, toothCount, turningInset],
  )
  const turningCandidates = solvabilityGraph.intersections
  const safeTurningIndex =
    turningCandidates.length === 0
      ? 0
      : clamp(selectedTurningIndex, 0, turningCandidates.length - 1)
  const selectedTurningCenter = useMemo(
    () => turningCandidates[safeTurningIndex] ?? turningCandidates[0] ?? { x: 0, y: 0, score: 0 },
    [safeTurningIndex, turningCandidates],
  )
  const outerPlanetCutter = useMemo(
    () => extractTurningProfile(planetOutline, selectedTurningCenter, 'outer'),
    [planetOutline, selectedTurningCenter],
  )
  const innerPlanetCutter = useMemo(
    () => extractTurningProfile(planetOutline, selectedTurningCenter, 'inner'),
    [planetOutline, selectedTurningCenter],
  )
  const averagePlanetRadius = useMemo(
    () => averagePerimeterRadius(outerPlanetCutter),
    [outerPlanetCutter],
  )
  const ringCarve = useMemo(
    () =>
      searchRingCarve({
        cutterOutline: planetOutline,
        turningCenter: selectedTurningCenter,
        planetToothCount: toothCount,
        planetCount,
        toothPitch,
        orbitRadius,
        ringThickness,
      }),
    [
      orbitRadius,
      planetOutline,
      planetCount,
      ringThickness,
      selectedTurningCenter,
      toothCount,
      toothPitch,
    ],
  )
  const mechanism = useMemo(
    () =>
      buildMechanismFromRingCarve({
        ringCarve,
        innerCutter: innerPlanetCutter,
        turningCenter: selectedTurningCenter,
        planetToothCount: toothCount,
        planetCount,
        toothPitch,
      }),
    [innerPlanetCutter, ringCarve, selectedTurningCenter, toothCount, planetCount, toothPitch],
  )
  const { carrierRadius, planetSpinFactor, sunSpinFactor, ringToothCount, sunToothCount } =
    mechanism
  const isSolvable = turningCandidates.length > 0 && mechanism.solvable
  const solvabilityCopy =
    turningCandidates.length === 0
      ? 'No internal turning center found for this shape.'
      : !ringCarve.solvable
        ? 'Step 1 ring carve fails the no-slip ring continuity checks.'
        : !mechanism.sunContinuous
          ? 'Step 2 sun carve does not resolve to one continuous valid curve.'
          : !mechanism.solvable
            ? 'This shape does not close as a printable planetary gear system.'
            : 'This shape currently passes the ring and sun no-slip checks.'
  const loopTurns = useMemo(
    () => findRepeatTurns([planetSpinFactor, sunSpinFactor]),
    [planetSpinFactor, sunSpinFactor],
  )
  const carrierAngle = progress * TAU * loopTurns
  const sunRotation = carrierAngle * sunSpinFactor
  const averageSunRadius = averagePerimeterRadius(mechanism.sunPitchCurve)
  const currentFrames = buildPlanetFrames(
    planetPitchCurve,
    planetOutline,
    selectedTurningCenter,
    carrierRadius,
    planetCount,
    carrierAngle,
    planetSpinFactor,
  )
  const rotatedSunOutline = mechanism.sunOutline.map((point) => rotatePoint(point, sunRotation))
  const rotatedSunPitch = mechanism.sunPitchCurve.map((point) => rotatePoint(point, sunRotation))
  const rotatedRingInnerBoundary = ringCarve.ringInnerBoundary
  const rotatedRingOuterBoundary = ringCarve.ringOuterBoundary
  const rotatedRingPitch = ringCarve.ringPitchCurve
  const ringBlankInner = ringCarve.baseInnerBlank
  const ringBlankOuter = ringCarve.baseOuterBlank

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
    const ringOuterPath = pointsToSmoothPath(rotatedRingOuterBoundary, CENTER, CENTER)
    const ringInnerPath = pointsToSmoothPath([...rotatedRingInnerBoundary].reverse(), CENTER, CENTER)
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
  const loadGearifySample = () => {
    setControlRadii(GEARIFY_SAMPLE_RADII)
    setBaseBias(DEFAULT_BASE_BIAS)
    setSmoothing(DEFAULT_SMOOTHING)
    setToothCount(DEFAULT_TOOTH_COUNT)
    setToothDepth(DEFAULT_TOOTH_DEPTH)
    setToothSharpness(DEFAULT_TOOTH_SHARPNESS)
    setPlanetCount(3)
    setOrbitRadius(DEFAULT_ORBIT_RADIUS)
    setRingThickness(DEFAULT_RING_THICKNESS)
    setTurningInset(DEFAULT_TURNING_INSET)
    setSelectedTurningIndex(0)
  }

  return (
    <div className="app-shell">
      <aside className="control-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Astronomer Prototype</p>
            <h1>Non-circular planetary gear designer</h1>
          </div>
          <div className="panel-header-actions">
            <button className="ghost-button" onClick={loadGearifySample}>
              Load sample
            </button>
            <button className="export-button" onClick={exportSvg}>
              Export SVG
            </button>
          </div>
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
            {solvabilityGraph.path.length > 1 ? (
              <path
                d={pointsToSmoothPath(solvabilityGraph.path, EDITOR_CENTER, EDITOR_CENTER)}
                className="solvability-path"
              />
            ) : null}
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
            Drag the handles to shape the planet. The green curve is the sampled solvability path;
            only its intersections where internal and external no-slip distances match are offered.
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
            <span>Build stages</span>
          </div>
          <div className="stage-toggle" role="tablist" aria-label="Generation steps">
            <button
              className={buildStage === 'ring' ? 'stage-button active' : 'stage-button'}
              onClick={() => setBuildStage('ring')}
            >
              1. Ring carve
            </button>
            <button
              className={buildStage === 'sun' ? 'stage-button active' : 'stage-button'}
              onClick={() => setBuildStage('sun')}
            >
              2. Sun carve
            </button>
            <button
              className={buildStage === 'assembly' ? 'stage-button active' : 'stage-button'}
              onClick={() => setBuildStage('assembly')}
            >
              3. Assembly
            </button>
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
          <div className="derived-metric">
            <span>Planet turns / orbit</span>
            <strong>{planetCount}</strong>
          </div>
          <div className="derived-metric">
            <span>Sun turns / orbit</span>
            <strong>{planetCount * 2}</strong>
          </div>
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
            <span>Solvability</span>
            <strong>{isSolvable ? 'Likely solvable' : 'Not solved'}</strong>
          </div>
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
          <div>
            <span>Pitch errors R/S</span>
            <strong>{`${Number.isFinite(ringCarve.ringPitchError) ? ringCarve.ringPitchError.toFixed(2) : '--'} / ${
              Number.isFinite(mechanism.sunPitchError) ? mechanism.sunPitchError.toFixed(2) : '--'
            }`}</strong>
          </div>
        </section>
      </aside>

      <main className="viewport-panel">
        <div className="viewport-header">
          <div>
            <p className="eyebrow">Live mechanism</p>
            <h2>
              {buildStage === 'ring'
                ? 'Step 1: ring carve'
                : buildStage === 'sun'
                  ? 'Step 2: sun carve'
                  : 'Step 3: full assembly'}
            </h2>
          </div>
          <p className="status-copy">
            {solvabilityCopy}
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
            d={`${
              buildStage === 'ring'
                ? `${pointsToSmoothPath(ringBlankOuter, CENTER, CENTER)} ${pointsToSmoothPath(
                    [...ringBlankInner].reverse(),
                    CENTER,
                    CENTER,
                  )}`
                : `${pointsToSmoothPath(rotatedRingOuterBoundary, CENTER, CENTER)} ${pointsToSmoothPath(
                    [...rotatedRingInnerBoundary].reverse(),
                    CENTER,
                    CENTER,
                  )}`
            }`}
            fill="url(#ring-fill)"
            fillRule="evenodd"
            stroke="#f7f2e8"
            strokeWidth="2.5"
          />

          {buildStage === 'ring' ? (
            <path
              d={pointsToSmoothPath(rotatedRingInnerBoundary, CENTER, CENTER)}
              fill="none"
              stroke="#f0b758"
              strokeWidth="2.1"
              opacity="0.9"
            />
          ) : null}

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
                d={pointsToSmoothPath(rotatedRingPitch, CENTER, CENTER)}
                fill="none"
                stroke="#8fd3ff"
                strokeWidth="1.4"
                opacity="0.34"
              />
              {buildStage !== 'ring' ? (
                <path
                  d={pointsToSmoothPath(rotatedSunPitch, CENTER, CENTER)}
                  fill="none"
                  stroke="#ffe082"
                  strokeWidth="1.3"
                  opacity="0.45"
                />
              ) : null}
            </>
          ) : null}

          {buildStage !== 'ring' && mechanism.sunContinuous ? (
            <path
              d={pointsToSmoothPath(rotatedSunOutline, CENTER, CENTER)}
              fill="#f0b758"
              stroke="#fff2d8"
              strokeWidth="1.5"
            />
          ) : null}

          {(buildStage === 'ring' ? currentFrames.outlineFrames.slice(0, 1) : orbitTrack).map((planet, index) => (
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
          <text x="28" y="94" className="viewport-label">{`Loop turns: ${loopTurns}`}</text>
        </svg>
      </main>
    </div>
  )
}

export default App
