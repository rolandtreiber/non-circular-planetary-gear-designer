import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

type Point = {
  x: number
  y: number
}

type TurningCandidate = Point & {
  score: number
  v: number
  externalDistance: number
  internalDistance: number
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
const DEFAULT_ANIMATION_SPEED = 0.02

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

function lerpPoint(a: Point, b: Point, t: number): Point {
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
  }
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

function pointInPolygon(point: Point, polygon: Point[]) {
  let inside = false

  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index, index += 1) {
    const current = polygon[index]
    const prior = polygon[previous]
    const intersects =
      current.y > point.y !== prior.y > point.y &&
      point.x < ((prior.x - current.x) * (point.y - current.y)) / ((prior.y - current.y) || 1e-9) + current.x

    if (intersects) {
      inside = !inside
    }
  }

  return inside
}

function distanceToOutline(point: Point, outline: Point[]) {
  let best = Infinity

  for (let index = 0; index < outline.length; index += 1) {
    best = Math.min(
      best,
      distancePointToSegment(point, outline[index], outline[(index + 1) % outline.length]),
    )
  }

  return best
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

function cubicBezierPoint(points: Point[], t: number) {
  const [p0, p1, p2, p3] = points
  const mt = 1 - t
  const mt2 = mt * mt
  const mt3 = mt2 * mt
  const t2 = t * t
  const t3 = t2 * t

  return {
    x: mt3 * p0.x + 3 * mt2 * t * p1.x + 3 * mt * t2 * p2.x + t3 * p3.x,
    y: mt3 * p0.y + 3 * mt2 * t * p1.y + 3 * mt * t2 * p2.y + t3 * p3.y,
  }
}

function sampleGuideCurve(points: Point[], sampleCount: number, outline: Point[], minInset: number) {
  if (points.length !== 4) {
    return []
  }

  const samples: Point[] = []
  for (let index = 0; index < sampleCount; index += 1) {
    const point = cubicBezierPoint(points, index / (sampleCount - 1))
    if (pointInPolygon(point, outline) && distanceToOutline(point, outline) >= minInset) {
      samples.push(point)
    }
  }

  return samples
}

function createDefaultGuideControlPoints(outline: Point[], minInset: number) {
  const autoPath = buildSolvabilityPath(outline, minInset)
  if (autoPath.length >= 4) {
    return [0, 0.33, 0.66, 1].map((t) => {
      const scaledIndex = t * (autoPath.length - 1)
      const lower = Math.floor(scaledIndex)
      const upper = Math.min(autoPath.length - 1, Math.ceil(scaledIndex))
      return lerpPoint(autoPath[lower], autoPath[upper], scaledIndex - lower)
    })
  }

  const { centroid, axis } = computePrincipalAxis(outline)
  return [-0.34, -0.12, 0.12, 0.34].map((offset) => ({
    x: centroid.x + axis.x * 120 * offset,
    y: centroid.y + axis.y * 120 * offset,
  }))
}

function smoothDistanceSeries(
  samples: Array<{
    point: Point
    v: number
    externalDistance: number
    internalDistance: number
    difference: number
  }>,
  iterations = 2,
) {
  let current = samples
  for (let step = 0; step < iterations; step += 1) {
    current = current.map((sample, index) => {
      const previous = current[Math.max(0, index - 1)]
      const next = current[Math.min(current.length - 1, index + 1)]
      const externalDistance =
        (previous.externalDistance + sample.externalDistance * 2 + next.externalDistance) / 4
      const internalDistance =
        (previous.internalDistance + sample.internalDistance * 2 + next.internalDistance) / 4
      return {
        ...sample,
        externalDistance,
        internalDistance,
        difference: externalDistance - internalDistance,
      }
    })
  }
  return current
}

function solveAffineFit(x: number[], y: number[]) {
  if (x.length === 0 || y.length === 0 || x.length !== y.length) {
    return { scale: 1, offset: 0 }
  }

  const meanX = x.reduce((acc, value) => acc + value, 0) / x.length
  const meanY = y.reduce((acc, value) => acc + value, 0) / y.length

  let covariance = 0
  let variance = 0
  for (let index = 0; index < x.length; index += 1) {
    const dx = x[index] - meanX
    covariance += dx * (y[index] - meanY)
    variance += dx * dx
  }

  const rawScale = variance > 1e-9 ? covariance / variance : 1
  const scale = clamp(rawScale, 0.25, 4.0)
  const offset = meanY - scale * meanX
  return { scale, offset }
}

function coupledFitDistanceSeries(
  samples: Array<{
    point: Point
    v: number
    externalDistance: number
    internalDistance: number
    difference: number
  }>,
  iterations = 8,
) {
  let fitted = samples.map((sample) => ({ ...sample }))
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const affine = solveAffineFit(
      fitted.map((sample) => sample.externalDistance),
      fitted.map((sample) => sample.internalDistance),
    )
    fitted = fitted.map((sample) => {
      const externalDistance = sample.externalDistance * affine.scale + affine.offset
      return {
        ...sample,
        externalDistance,
        difference: externalDistance - sample.internalDistance,
      }
    })
    fitted = smoothDistanceSeries(fitted, 1)
  }

  return fitted
}

function interpolateDistanceSampleAtV(
  samples: Array<{
    point: Point
    v: number
    externalDistance: number
    internalDistance: number
    difference: number
  }>,
  v: number,
) {
  if (samples.length === 0) {
    return { externalDistance: 0, internalDistance: 0 }
  }
  if (v <= samples[0].v) {
    return {
      externalDistance: samples[0].externalDistance,
      internalDistance: samples[0].internalDistance,
    }
  }
  if (v >= samples[samples.length - 1].v) {
    return {
      externalDistance: samples[samples.length - 1].externalDistance,
      internalDistance: samples[samples.length - 1].internalDistance,
    }
  }

  for (let index = 0; index + 1 < samples.length; index += 1) {
    const a = samples[index]
    const b = samples[index + 1]
    if (v < a.v || v > b.v) {
      continue
    }
    const span = Math.max(1e-9, b.v - a.v)
    const t = (v - a.v) / span
    return {
      externalDistance: a.externalDistance + (b.externalDistance - a.externalDistance) * t,
      internalDistance: a.internalDistance + (b.internalDistance - a.internalDistance) * t,
    }
  }

  return {
    externalDistance: samples[0].externalDistance,
    internalDistance: samples[0].internalDistance,
  }
}

function interpolatePointOnPath(path: Point[], v: number) {
  if (path.length === 0) {
    return { x: 0, y: 0 }
  }
  if (path.length === 1 || v <= 0) {
    return path[0]
  }
  if (v >= 1) {
    return path[path.length - 1]
  }

  const scaledIndex = v * (path.length - 1)
  const lower = Math.floor(scaledIndex)
  const upper = Math.min(path.length - 1, Math.ceil(scaledIndex))
  const t = scaledIndex - lower
  return {
    x: path[lower].x + (path[upper].x - path[lower].x) * t,
    y: path[lower].y + (path[upper].y - path[lower].y) * t,
  }
}

function findDisplayedIntersectionNearV(
  samples: Array<{
    point: Point
    v: number
    externalDistance: number
    internalDistance: number
    difference: number
  }>,
  targetV: number,
) {
  if (samples.length === 0) {
    return { v: targetV, distance: 0 }
  }

  let bestSegmentIndex = 0
  let bestSegmentDistance = Number.POSITIVE_INFINITY

  for (let index = 0; index + 1 < samples.length; index += 1) {
    const a = samples[index]
    const b = samples[index + 1]
    const segmentCenter = (a.v + b.v) * 0.5
    const segmentDistance = Math.abs(segmentCenter - targetV)
    const da = a.externalDistance - a.internalDistance
    const db = b.externalDistance - b.internalDistance
    if (da * db <= 0 && segmentDistance < bestSegmentDistance) {
      bestSegmentDistance = segmentDistance
      bestSegmentIndex = index
    }
  }

  if (Number.isFinite(bestSegmentDistance)) {
    const a = samples[bestSegmentIndex]
    const b = samples[bestSegmentIndex + 1]
    const da = a.externalDistance - a.internalDistance
    const db = b.externalDistance - b.internalDistance
    const blend = Math.abs(da) / (Math.abs(da) + Math.abs(db) || 1)
    const v = a.v + (b.v - a.v) * blend
    const externalDistance = a.externalDistance + (b.externalDistance - a.externalDistance) * blend
    const internalDistance = a.internalDistance + (b.internalDistance - a.internalDistance) * blend
    return { v, distance: (externalDistance + internalDistance) * 0.5 }
  }

  let nearest = samples[0]
  let nearestError = Math.abs(samples[0].v - targetV) + Math.abs(samples[0].externalDistance - samples[0].internalDistance)
  for (let index = 1; index < samples.length; index += 1) {
    const sample = samples[index]
    const error = Math.abs(sample.v - targetV) + Math.abs(sample.externalDistance - sample.internalDistance)
    if (error < nearestError) {
      nearestError = error
      nearest = sample
    }
  }

  return { v: nearest.v, distance: (nearest.externalDistance + nearest.internalDistance) * 0.5 }
}

function findSolvableTurningCandidates(
  baseCurve: Point[],
  guideCurveControls: Point[],
  externalRatio: number,
  internalRatio: number,
  minInset: number,
) {
  const path = sampleGuideCurve(guideCurveControls, 64, baseCurve, minInset)
  if (path.length < 2) {
    return { path: [], samples: [], intersections: [] } satisfies SolvabilityGraph
  }

  const rawSamples = path.map((point, index) => {
    const outerProfile = extractTurningProfile(baseCurve, point, 'outer')
    const innerProfile = extractTurningProfile(baseCurve, point, 'inner')
    const outerRadius = averagePerimeterRadius(outerProfile)
    const innerRadius = averagePerimeterRadius(innerProfile)
    const externalDistance = outerRadius * externalRatio
    const internalDistance = innerRadius * internalRatio

    return {
      point,
      v: path.length === 1 ? 0 : index / (path.length - 1),
      externalDistance,
      internalDistance,
      difference: externalDistance - internalDistance,
    }
  })
  const physicalSamples = smoothDistanceSeries(rawSamples, 2)
  const samples = coupledFitDistanceSeries(physicalSamples, 10)

  const intersections: TurningCandidate[] = []

  for (let index = 0; index + 1 < physicalSamples.length; index += 1) {
    const current = physicalSamples[index]
    const next = physicalSamples[index + 1]
    const currentDifference = current.difference
    const nextDifference = next.difference

    if (Math.abs(currentDifference) < 1e-3) {
      const displayAtV = interpolateDistanceSampleAtV(samples, current.v)
      intersections.push({
        ...current.point,
        score: 1 / (1 + Math.abs(currentDifference)),
        v: current.v,
        externalDistance: displayAtV.externalDistance,
        internalDistance: displayAtV.internalDistance,
      })
      continue
    }

    if (currentDifference * nextDifference > 0) {
      continue
    }

    const blend = Math.abs(currentDifference) / (Math.abs(currentDifference) + Math.abs(nextDifference) || 1)
    const intersectionV = current.v + (next.v - current.v) * blend
    const displayAtV = interpolateDistanceSampleAtV(samples, intersectionV)
    intersections.push({
      x: current.point.x + (next.point.x - current.point.x) * blend,
      y: current.point.y + (next.point.y - current.point.y) * blend,
      score: 1 / (1 + Math.abs(currentDifference) + Math.abs(nextDifference)),
      v: intersectionV,
      externalDistance: displayAtV.externalDistance,
      internalDistance: displayAtV.internalDistance,
    })
  }

  if (intersections.length === 0) {
    for (let index = 0; index + 1 < samples.length; index += 1) {
      const current = samples[index]
      const next = samples[index + 1]
      const currentDifference = current.externalDistance - current.internalDistance
      const nextDifference = next.externalDistance - next.internalDistance
      if (currentDifference * nextDifference > 0) {
        continue
      }
      const blend = Math.abs(currentDifference) / (Math.abs(currentDifference) + Math.abs(nextDifference) || 1)
      const v = current.v + (next.v - current.v) * blend
      const displayAtV = interpolateDistanceSampleAtV(samples, v)
      const point = interpolatePointOnPath(path, v)
      intersections.push({
        x: point.x,
        y: point.y,
        score: 0.25,
        v,
        externalDistance: displayAtV.externalDistance,
        internalDistance: displayAtV.internalDistance,
      })
    }
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

function pointsToOpenSmoothPath(points: Point[], cx: number, cy: number) {
  if (points.length === 0) {
    return ''
  }
  if (points.length === 1) {
    return `M ${points[0].x + cx} ${points[0].y + cy}`
  }
  if (points.length < 3) {
    return points
      .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x + cx} ${point.y + cy}`)
      .join(' ')
  }

  let path = `M ${points[0].x + cx} ${points[0].y + cy}`

  for (let index = 0; index < points.length - 1; index += 1) {
    const p0 = points[Math.max(0, index - 1)]
    const p1 = points[index]
    const p2 = points[index + 1]
    const p3 = points[Math.min(points.length - 1, index + 2)]
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

  return path
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
  const length = radial.length
  const knownIndices = radial
    .map((value, index) => (Number.isFinite(value) ? index : -1))
    .filter((index) => index >= 0)

  if (knownIndices.length === 0) {
    return radial.map(() => fallback)
  }

  if (knownIndices.length === 1) {
    return radial.map(() => radial[knownIndices[0]] ?? fallback)
  }

  const filled = [...radial]

  for (let knownIndex = 0; knownIndex < knownIndices.length; knownIndex += 1) {
    const startIndex = knownIndices[knownIndex]
    const endIndex = knownIndices[(knownIndex + 1) % knownIndices.length]
    const startValue = filled[startIndex]
    const endValue = filled[endIndex]

    let steps = endIndex - startIndex
    if (steps <= 0) {
      steps += length
    }

    for (let step = 1; step < steps; step += 1) {
      const index = (startIndex + step) % length
      if (Number.isFinite(filled[index])) {
        continue
      }

      const blend = step / steps
      filled[index] = startValue + (endValue - startValue) * blend
    }
  }

  return filled.map((value) => (Number.isFinite(value) ? value : fallback))
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

function buildRadialLookup(points: Point[], bins: number, mode: 'inner' | 'outer') {
  const radial = Array.from({ length: bins }, () =>
    mode === 'outer' ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY,
  )

  points.forEach((point) => {
    const angle = normalizeAngle(Math.atan2(point.y, point.x))
    const radius = Math.hypot(point.x, point.y)
    const bin = Math.floor((angle / TAU) * bins) % bins
    radial[bin] =
      mode === 'outer'
        ? Math.max(radial[bin], radius)
        : Math.min(radial[bin], radius)
  })

  const known = radial.filter((value) => Number.isFinite(value))
  const fallback =
    mode === 'outer'
      ? Math.max(...known, 1)
      : Math.min(...known, 1)

  return fillRadialGaps(
    radial.map((value) => (Number.isFinite(value) ? value : fallback)),
    fallback,
  )
}

function sampleLookupRadial(radial: number[], angle: number) {
  const position = (normalizeAngle(angle) / TAU) * radial.length
  const lower = Math.floor(position) % radial.length
  const upper = (lower + 1) % radial.length
  const blend = position - Math.floor(position)
  return radial[lower] + (radial[upper] - radial[lower]) * blend
}

function measureSunContactError(
  frames: Point[][],
  sunRadialLookup: number[],
  sunSpinFactor: number,
  phaseOffset: number,
) {
  let overlap = 0
  let clearance = 0
  let minGapDrift = 0

  frames.forEach((frame, frameIndex) => {
    const carrierAngle = (frameIndex / Math.max(frames.length, 1)) * TAU
    const sunRotation = carrierAngle * sunSpinFactor + phaseOffset
    let frameMinGap = Number.POSITIVE_INFINITY

    frame.forEach((point) => {
      const radius = Math.hypot(point.x, point.y)
      // Query sun radius in the sun's local frame for this timestep.
      const sunRadius = sampleLookupRadial(sunRadialLookup, Math.atan2(point.y, point.x) - sunRotation)
      const gap = radius - sunRadius
      frameMinGap = Math.min(frameMinGap, gap)
    })
    overlap += Math.max(0, -frameMinGap)
    clearance += Math.max(0, frameMinGap)
    minGapDrift += frameMinGap * frameMinGap
  })

  const sampleCount = Math.max(frames.length, 1)
  return {
    overlap: overlap / sampleCount,
    clearance: clearance / sampleCount,
    minGapRms: Math.sqrt(minGapDrift / sampleCount),
  }
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

function expandFramesForPlanets(frames: Point[][], planetCount: number) {
  if (planetCount <= 1) {
    return frames
  }

  return frames.map((frame) => {
    const points: Point[] = []
    for (let planetIndex = 0; planetIndex < planetCount; planetIndex += 1) {
      const offset = (planetIndex / planetCount) * TAU
      frame.forEach((point) => {
        points.push(rotatePoint(point, offset))
      })
    }
    return points
  })
}

type MechanismGeometry = {
  carrierRadius: number
  planetSpinFactor: number
  sunSpinFactor: number
  sunPhaseOffset: number
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
  sunToothCount: number
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

type BuildStage = 'diagram' | 'ring' | 'sun' | 'assembly'

type AppliedInputs = {
  controlRadii: number[]
  baseBias: number
  smoothing: number
  toothCount: number
  ringToothCount: number
  sunToothCount: number
  toothDepth: number
  toothSharpness: number
  planetCount: number
  planetTurnsPerOrbit: number
  sunTurnsPerOrbit: number
  orbitRadius: number
  ringThickness: number
  turningInset: number
  guideControlPoints: Point[]
}

function areNumberArraysEqual(a: number[], b: number[]) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) < 1e-9)
}

function arePointArraysEqual(a: Point[], b: Point[]) {
  return (
    a.length === b.length &&
    a.every((value, index) => Math.abs(value.x - b[index].x) < 1e-9 && Math.abs(value.y - b[index].y) < 1e-9)
  )
}

function searchRingCarve({
  cutterPitchCurve,
  turningCenter,
  ringToothCount,
  sunToothCount,
  planetTurnsPerOrbit,
  toothPitch,
  toothDepth,
  toothSharpness,
  orbitRadius,
  ringThickness,
}: {
  cutterPitchCurve: Point[]
  turningCenter: Point
  ringToothCount: number
  sunToothCount: number
  planetTurnsPerOrbit: number
  toothPitch: number
  toothDepth: number
  toothSharpness: number
  orbitRadius: number
  ringThickness: number
}): RingCarveResult {
  const averagePlanetRadius = averagePerimeterRadius(cutterPitchCurve)
  const rollingPlanetRadius = averagePerimeterRadius(
    cutterPitchCurve.map((point) => ({ x: point.x - turningCenter.x, y: point.y - turningCenter.y })),
  )
  const sampleCount = 240
  // No-slip kinematic anchor:
  // |planetSpinFactor| ~= carrierRadius / rollingPlanetRadius
  const targetCarrierRadius = Math.max(
    rollingPlanetRadius * 1.1,
    Math.abs(planetTurnsPerOrbit) * rollingPlanetRadius,
  )
  const radiusSpread = Math.max(18, targetCarrierRadius * 0.18)
  const radiusCandidates = Array.from({ length: 21 }, (_, index) =>
    clamp(
      targetCarrierRadius - radiusSpread + ((index * 2 * radiusSpread) / 20),
      rollingPlanetRadius * 1.1,
      rollingPlanetRadius * 14,
    ),
  )
  const planetSpinFactor = -Math.max(0.25, planetTurnsPerOrbit)

  let best: (RingCarveResult & { score: number }) | null = null

  radiusCandidates.forEach((carrierRadius) => {
    const ringFrames = samplePlanetMotion(
      cutterPitchCurve,
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
      2,
    )
    const ringPitchCurve = resampleClosedPolyline(
      smoothRadialEnvelope(ringEnvelope, 1),
      Math.max(720, ringToothCount * 48),
    )
    const ringPitchLength = computeArcLengths(ringPitchCurve).total
    const requestedRingToothCount = Math.max(6, Math.round(ringToothCount))
    const ringToothCountTarget = Math.max(6, Math.round(ringPitchLength / toothPitch))
    const ringInnerBoundary = createToothedOutline(
      ringPitchCurve,
      ringToothCountTarget,
      toothDepth,
      toothSharpness,
      true,
    )
    const ringPitchError = Math.abs(ringPitchLength / toothPitch - ringToothCountTarget)
    const ringToothPreferenceError = Math.abs(ringToothCountTarget - requestedRingToothCount) / requestedRingToothCount
    const sunToothCountTarget = Math.max(6, Math.round(sunToothCount))
    const targetSunPitchLength = toothPitch * sunToothCountTarget
    const targetSunRadius = targetSunPitchLength / TAU

    const continuityPenalty = Math.abs(computeSignedArea(ringPitchCurve)) < 1 ? 1000 : 0
    const kinematicPenalty =
      Math.abs(Math.abs(planetSpinFactor) - carrierRadius / Math.max(rollingPlanetRadius, 1e-6)) * 5.5
    const orbitPreferencePenalty = Math.abs(carrierRadius - orbitRadius) * 0.03
    const score =
      ringPitchError * 10 +
      ringToothPreferenceError * 1.8 +
      kinematicPenalty +
      orbitPreferencePenalty +
      Math.abs(carrierRadius - targetCarrierRadius) * 0.08 +
      Math.abs(averagePerimeterRadius(ringPitchCurve) - (carrierRadius + rollingPlanetRadius)) * 0.04 +
      continuityPenalty

    if (!best || score < best.score) {
      best = {
        carrierRadius,
        planetSpinFactor,
        ringToothCount: ringToothCountTarget,
        sunToothCount: sunToothCountTarget,
        ringInnerBoundary,
        ringOuterBoundary: createRingShell(ringInnerBoundary, ringThickness),
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

  const fallbackRing = createRingShell(cutterPitchCurve, ringThickness * 2.2)
  const fallbackRadius = averagePerimeterRadius(fallbackRing)
  const fallbackRingToothCount = Math.max(6, Math.round((TAU * fallbackRadius) / toothPitch))
  const fallbackSunToothCount = Math.max(6, Math.round(sunToothCount))
  return {
    carrierRadius: targetCarrierRadius || orbitRadius,
    planetSpinFactor: -Math.max(0.25, planetTurnsPerOrbit),
    ringToothCount: fallbackRingToothCount,
    sunToothCount: fallbackSunToothCount,
    ringInnerBoundary: fallbackRing,
    ringOuterBoundary: createRingShell(fallbackRing, ringThickness),
    ringPitchCurve: fallbackRing,
    ringPitchError: Infinity,
    baseInnerBlank: createCirclePoints(fallbackRadius, 360),
    baseOuterBlank: createCirclePoints(fallbackRadius + ringThickness, 360),
    targetSunPitchLength: fallbackSunToothCount * toothPitch,
    targetSunRadius: (fallbackSunToothCount * toothPitch) / TAU,
    solvable: false,
  }
}

function buildMechanismFromRingCarve({
  ringCarve,
  innerPitchCutter,
  innerToothedCutter,
  turningCenter,
  planetCount,
  planetToothCount,
  sunTurnsPerOrbit,
  toothDepth,
  toothSharpness,
}: {
  ringCarve: RingCarveResult
  innerPitchCutter: Point[]
  innerToothedCutter: Point[]
  turningCenter: Point
  planetCount: number
  planetToothCount: number
  sunTurnsPerOrbit: number
  toothDepth: number
  toothSharpness: number
}): MechanismGeometry {
  const sampleCount = 1080
  const sunToothCount = Math.max(6, Math.round(planetToothCount * Math.max(1, Math.abs(sunTurnsPerOrbit))))
  const sunSpinFactor = Math.max(0.25, sunTurnsPerOrbit)
  const sunFrames = samplePlanetMotion(
    innerPitchCutter,
    turningCenter,
    ringCarve.carrierRadius,
    ringCarve.planetSpinFactor,
    sampleCount,
  ).map((frame, step) => {
    const carrierAngle = (step / sampleCount) * TAU
    const sunRotation = carrierAngle * sunSpinFactor
    return frame.map((point) => rotatePoint(point, -sunRotation))
  })
  const sunEnvelope = buildConjugateEnvelope(
    sunFrames,
    'inner',
    Math.max(ringCarve.targetSunRadius * 1.12, averagePerimeterRadius(innerPitchCutter)),
    5760,
    5,
  )
  const baseSunPitchCurve = resampleClosedPolyline(
    sunEnvelope,
    Math.max(960, sunToothCount * 72),
  )
  const toothedFrames = samplePlanetMotion(
    innerToothedCutter,
    turningCenter,
    ringCarve.carrierRadius,
    ringCarve.planetSpinFactor,
    Math.max(240, Math.floor(sampleCount / 2)),
  )
  const expandedToothedFrames = expandFramesForPlanets(toothedFrames, planetCount)
  let bestSunPitchCurve = baseSunPitchCurve
  let bestSunOutline = createToothedOutline(
    baseSunPitchCurve,
    sunToothCount,
    toothDepth,
    toothSharpness,
    false,
  )
  let sunPhaseOffset = 0
  let bestCombinedScore = Number.POSITIVE_INFINITY
  const phaseCandidates = Math.max(96, sunToothCount * 8)
  const scaleCandidates = Array.from({ length: 17 }, (_, index) => 0.8 + index * 0.03)

  scaleCandidates.forEach((scale) => {
    const scaledPitch = baseSunPitchCurve.map((point) => ({ x: point.x * scale, y: point.y * scale }))
    const scaledOutline = createToothedOutline(
      scaledPitch,
      sunToothCount,
      toothDepth,
      toothSharpness,
      false,
    )
    const sunRadialLookup = buildRadialLookup(scaledOutline, 4096, 'outer')
    const averageRadius = averagePerimeterRadius(scaledPitch)
    let localBestPhase = 0
    let localBestScore = Number.POSITIVE_INFINITY

    for (let phaseIndex = 0; phaseIndex < phaseCandidates; phaseIndex += 1) {
      const candidateOffset = (phaseIndex / phaseCandidates) * TAU
      const contact = measureSunContactError(
        expandedToothedFrames,
        sunRadialLookup,
        sunSpinFactor,
        candidateOffset,
      )
      const phaseScore =
        contact.overlap * 850 +
        contact.clearance * 6 +
        contact.minGapRms * 20 -
        averageRadius * 0.03

      if (phaseScore < localBestScore) {
        localBestScore = phaseScore
        localBestPhase = candidateOffset
      }
    }

    if (localBestScore < bestCombinedScore) {
      bestCombinedScore = localBestScore
      bestSunPitchCurve = scaledPitch
      bestSunOutline = scaledOutline
      sunPhaseOffset = localBestPhase
    }
  })

  const sunPitchCurve = bestSunPitchCurve
  const sunOutline = bestSunOutline
  const sunPitchLength = computeArcLengths(sunPitchCurve).total
  const sunPitchError = Math.abs(sunPitchLength / Math.max(1e-6, ringCarve.targetSunPitchLength / ringCarve.sunToothCount) - sunToothCount)
  const sunAverageRadius = averagePerimeterRadius(sunPitchCurve)
  const sunContinuous =
    Math.abs(computeSignedArea(sunPitchCurve)) > 1 &&
    maxEdgeLength(sunPitchCurve) / Math.max(averageEdgeLength(sunPitchCurve), 1e-6) < 2.3 &&
    maxRadialJump(sunPitchCurve) / Math.max(sunAverageRadius, 1e-6) < 0.16 &&
    sunAverageRadius > averagePerimeterRadius(innerPitchCutter) * 0.7

  return {
    carrierRadius: ringCarve.carrierRadius,
    planetSpinFactor: ringCarve.planetSpinFactor,
    sunSpinFactor,
    sunPhaseOffset,
    ringToothCount: ringCarve.ringToothCount,
    sunToothCount,
    ringInnerBoundary: ringCarve.ringInnerBoundary,
    ringOuterBoundary: ringCarve.ringOuterBoundary,
    ringPitchCurve: ringCarve.ringPitchCurve,
    sunOutline,
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
  const [ringToothCountInput, setRingToothCountInput] = useState(DEFAULT_TOOTH_COUNT * 3)
  const [sunToothCountInput, setSunToothCountInput] = useState(DEFAULT_TOOTH_COUNT * 2)
  const [toothDepth, setToothDepth] = useState(DEFAULT_TOOTH_DEPTH)
  const [toothSharpness, setToothSharpness] = useState(DEFAULT_TOOTH_SHARPNESS)
  const [planetCount, setPlanetCount] = useState(3)
  const [planetTurnsPerOrbit, setPlanetTurnsPerOrbit] = useState(6)
  const [sunTurnsPerOrbit, setSunTurnsPerOrbit] = useState(3)
  const [orbitRadius, setOrbitRadius] = useState(DEFAULT_ORBIT_RADIUS)
  const [ringThickness, setRingThickness] = useState(DEFAULT_RING_THICKNESS)
  const [turningInset, setTurningInset] = useState(DEFAULT_TURNING_INSET)
  const [selectedTurningIndex, setSelectedTurningIndex] = useState(0)
  const [buildStage, setBuildStage] = useState<BuildStage>('diagram')
  const [showPitch, setShowPitch] = useState(true)
  const [showCenters, setShowCenters] = useState(true)
  const [showTracks, setShowTracks] = useState(true)
  const [animationSpeed, setAnimationSpeed] = useState(DEFAULT_ANIMATION_SPEED)
  const [progress, setProgress] = useState(0)
  const [dragIndex, setDragIndex] = useState<number | null>(null)
  const [guideDragIndex, setGuideDragIndex] = useState<number | null>(null)
  const [appliedInputs, setAppliedInputs] = useState<AppliedInputs>({
    controlRadii: GEARIFY_SAMPLE_RADII,
    baseBias: DEFAULT_BASE_BIAS,
    smoothing: DEFAULT_SMOOTHING,
    toothCount: DEFAULT_TOOTH_COUNT,
    ringToothCount: DEFAULT_TOOTH_COUNT * 3,
    sunToothCount: DEFAULT_TOOTH_COUNT * 2,
    toothDepth: DEFAULT_TOOTH_DEPTH,
    toothSharpness: DEFAULT_TOOTH_SHARPNESS,
    planetCount: 3,
    planetTurnsPerOrbit: 6,
    sunTurnsPerOrbit: 3,
    orbitRadius: DEFAULT_ORBIT_RADIUS,
    ringThickness: DEFAULT_RING_THICKNESS,
    turningInset: DEFAULT_TURNING_INSET,
    guideControlPoints: [],
  })

  const editorRef = useRef<SVGSVGElement | null>(null)

  const samplesPerTooth = 40
  const livePitchCurve = useMemo(
    () => createPitchCurve(controlRadii, baseBias, smoothing),
    [controlRadii, baseBias, smoothing],
  )
  const livePlanetPitchCurve = useMemo(
    () => resampleClosedPolyline(livePitchCurve, toothCount * samplesPerTooth),
    [livePitchCurve, toothCount],
  )
  const livePlanetOutline = useMemo(
    () =>
      createToothedOutline(livePlanetPitchCurve, toothCount, toothDepth, toothSharpness),
    [livePlanetPitchCurve, toothCount, toothDepth, toothSharpness],
  )
  const defaultGuideControlPoints = useMemo(
    () => createDefaultGuideControlPoints(livePlanetPitchCurve, turningInset),
    [livePlanetPitchCurve, turningInset],
  )
  const [guideControlPoints, setGuideControlPoints] = useState<Point[]>([])
  const activeGuideControlPoints =
    guideControlPoints.length === 4 ? guideControlPoints : defaultGuideControlPoints

  const hasPendingSolveChanges = useMemo(
    () =>
      !areNumberArraysEqual(controlRadii, appliedInputs.controlRadii) ||
      baseBias !== appliedInputs.baseBias ||
      smoothing !== appliedInputs.smoothing ||
      toothCount !== appliedInputs.toothCount ||
      ringToothCountInput !== appliedInputs.ringToothCount ||
      sunToothCountInput !== appliedInputs.sunToothCount ||
      toothDepth !== appliedInputs.toothDepth ||
      toothSharpness !== appliedInputs.toothSharpness ||
      planetCount !== appliedInputs.planetCount ||
      planetTurnsPerOrbit !== appliedInputs.planetTurnsPerOrbit ||
      sunTurnsPerOrbit !== appliedInputs.sunTurnsPerOrbit ||
      orbitRadius !== appliedInputs.orbitRadius ||
      ringThickness !== appliedInputs.ringThickness ||
      turningInset !== appliedInputs.turningInset ||
      !arePointArraysEqual(guideControlPoints, appliedInputs.guideControlPoints),
    [
      appliedInputs,
      baseBias,
      controlRadii,
      guideControlPoints,
      orbitRadius,
      planetCount,
      planetTurnsPerOrbit,
      ringThickness,
      ringToothCountInput,
      smoothing,
      sunToothCountInput,
      sunTurnsPerOrbit,
      toothCount,
      toothDepth,
      toothSharpness,
      turningInset,
    ],
  )

  const applyRecalculation = () => {
    setAppliedInputs({
      controlRadii: [...controlRadii],
      baseBias,
      smoothing,
      toothCount,
      ringToothCount: ringToothCountInput,
      sunToothCount: sunToothCountInput,
      toothDepth,
      toothSharpness,
      planetCount,
      planetTurnsPerOrbit,
      sunTurnsPerOrbit,
      orbitRadius,
      ringThickness,
      turningInset,
      guideControlPoints: guideControlPoints.map((point) => ({ ...point })),
    })
    setSelectedTurningIndex(0)
  }

  const pitchCurve = useMemo(
    () => createPitchCurve(appliedInputs.controlRadii, appliedInputs.baseBias, appliedInputs.smoothing),
    [appliedInputs.baseBias, appliedInputs.controlRadii, appliedInputs.smoothing],
  )
  const derivedSunToothCount = useMemo(
    () => Math.max(6, Math.round(appliedInputs.toothCount * Math.max(1, Math.abs(appliedInputs.sunTurnsPerOrbit)))),
    [appliedInputs.sunTurnsPerOrbit, appliedInputs.toothCount],
  )
  const basePitchLength = useMemo(() => computeArcLengths(pitchCurve).total, [pitchCurve])
  const toothPitch = basePitchLength / appliedInputs.toothCount
  const planetPitchCurve = useMemo(
    () => resampleClosedPolyline(pitchCurve, appliedInputs.toothCount * samplesPerTooth),
    [appliedInputs.toothCount, pitchCurve],
  )
  const planetOutline = useMemo(
    () =>
      createToothedOutline(planetPitchCurve, appliedInputs.toothCount, appliedInputs.toothDepth, appliedInputs.toothSharpness),
    [appliedInputs.toothCount, appliedInputs.toothDepth, appliedInputs.toothSharpness, planetPitchCurve],
  )
  const appliedGuideControls = useMemo(
    () =>
      appliedInputs.guideControlPoints.length === 4
        ? appliedInputs.guideControlPoints
        : createDefaultGuideControlPoints(planetPitchCurve, appliedInputs.turningInset),
    [appliedInputs.guideControlPoints, appliedInputs.turningInset, planetPitchCurve],
  )

  const solvabilityGraph = useMemo(
    () =>
      findSolvableTurningCandidates(
        planetPitchCurve,
        appliedGuideControls,
        appliedInputs.planetTurnsPerOrbit,
        appliedInputs.sunTurnsPerOrbit,
        appliedInputs.turningInset,
      ),
    [
      appliedGuideControls,
      appliedInputs.planetTurnsPerOrbit,
      appliedInputs.sunTurnsPerOrbit,
      appliedInputs.turningInset,
      planetPitchCurve,
    ],
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
  const innerPlanetPitchCutter = useMemo(
    () => extractTurningProfile(planetPitchCurve, selectedTurningCenter, 'inner'),
    [planetPitchCurve, selectedTurningCenter],
  )
  const innerPlanetToothedCutter = useMemo(
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
        cutterPitchCurve: planetPitchCurve,
        turningCenter: selectedTurningCenter,
        ringToothCount: appliedInputs.ringToothCount,
        sunToothCount: derivedSunToothCount,
        planetTurnsPerOrbit: appliedInputs.planetTurnsPerOrbit,
        toothPitch,
        toothDepth: appliedInputs.toothDepth,
        toothSharpness: appliedInputs.toothSharpness,
        orbitRadius: appliedInputs.orbitRadius,
        ringThickness: appliedInputs.ringThickness,
      }),
    [
      planetPitchCurve,
      appliedInputs.orbitRadius,
      appliedInputs.planetTurnsPerOrbit,
      appliedInputs.ringThickness,
      appliedInputs.ringToothCount,
      derivedSunToothCount,
      appliedInputs.toothDepth,
      appliedInputs.toothSharpness,
      selectedTurningCenter,
      toothPitch,
    ],
  )
  const mechanism = useMemo(
    () =>
      buildMechanismFromRingCarve({
        ringCarve,
        innerPitchCutter: innerPlanetPitchCutter,
        innerToothedCutter: innerPlanetToothedCutter,
        turningCenter: selectedTurningCenter,
        planetCount: appliedInputs.planetCount,
        planetToothCount: appliedInputs.toothCount,
        sunTurnsPerOrbit: appliedInputs.sunTurnsPerOrbit,
        toothDepth: appliedInputs.toothDepth,
        toothSharpness: appliedInputs.toothSharpness,
      }),
    [
      innerPlanetPitchCutter,
      innerPlanetToothedCutter,
      ringCarve,
      selectedTurningCenter,
      appliedInputs.planetCount,
      appliedInputs.toothCount,
      appliedInputs.sunTurnsPerOrbit,
      appliedInputs.toothDepth,
      appliedInputs.toothSharpness,
    ],
  )
  const { carrierRadius, planetSpinFactor, sunSpinFactor, sunPhaseOffset, ringToothCount, sunToothCount } =
    mechanism
  const isSolvable = turningCandidates.length > 0 && mechanism.solvable
  const solvabilityCopy =
    hasPendingSolveChanges
      ? 'Parameters changed. Click Recalculate to run heavy geometry solve.'
      : turningCandidates.length === 0
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
  const sunRotation = carrierAngle * sunSpinFactor + sunPhaseOffset
  const averageSunRadius = averagePerimeterRadius(mechanism.sunPitchCurve)
  const currentFrames = buildPlanetFrames(
    planetPitchCurve,
    planetOutline,
    selectedTurningCenter,
    carrierRadius,
    appliedInputs.planetCount,
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
  const previewScale = useMemo(() => {
    if (buildStage === 'diagram') {
      return 1
    }

    const clouds: Point[][] = []
    if (buildStage === 'ring') {
      clouds.push(ringBlankOuter, ringBlankInner, rotatedRingInnerBoundary)
      clouds.push(currentFrames.outlineFrames[0] ?? [])
    } else {
      clouds.push(rotatedRingOuterBoundary, rotatedRingInnerBoundary)
      if (mechanism.sunContinuous) {
        clouds.push(rotatedSunOutline)
      }
      clouds.push(...currentFrames.outlineFrames)
    }

    let maxRadius = 1
    for (const cloud of clouds) {
      for (const point of cloud) {
        maxRadius = Math.max(maxRadius, Math.hypot(point.x, point.y))
      }
    }

    const targetRadius = VIEWBOX_SIZE * 0.43
    return clamp(targetRadius / maxRadius, 0.32, 1)
  }, [
    buildStage,
    currentFrames.outlineFrames,
    mechanism.sunContinuous,
    ringBlankInner,
    ringBlankOuter,
    rotatedRingInnerBoundary,
    rotatedRingOuterBoundary,
    rotatedSunOutline,
  ])
  const graphBounds = useMemo(() => {
    const values = solvabilityGraph.samples.flatMap((sample) => [
      sample.externalDistance,
      sample.internalDistance,
    ])
    const minDistance = values.length > 0 ? Math.min(...values) : 0
    const maxDistance = values.length > 0 ? Math.max(...values) : 1
    const padding = Math.max(8, (maxDistance - minDistance) * 0.08)
    return {
      minDistance: minDistance - padding,
      maxDistance: maxDistance + padding,
    }
  }, [solvabilityGraph.samples])

  const mapGraphPoint = (v: number, d: number): Point => {
    const left = 96
    const right = VIEWBOX_SIZE - 72
    const top = 84
    const bottom = VIEWBOX_SIZE - 92
    const width = right - left
    const height = bottom - top
    const vValue = left + width * v
    const normalizedDistance =
      (d - graphBounds.minDistance) / (graphBounds.maxDistance - graphBounds.minDistance || 1)
    const dValue = bottom - normalizedDistance * height
    return { x: vValue, y: dValue }
  }

  const diagramExternalCurve = solvabilityGraph.samples.map((sample) =>
    mapGraphPoint(sample.v, sample.externalDistance),
  )
  const diagramInternalCurve = solvabilityGraph.samples.map((sample) =>
    mapGraphPoint(sample.v, sample.internalDistance),
  )
  const diagramIntersections = turningCandidates.map((candidate) => {
    const snapped = findDisplayedIntersectionNearV(solvabilityGraph.samples, candidate.v)
    return {
      candidate,
      point: mapGraphPoint(snapped.v, snapped.distance),
    }
  })

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

  useEffect(() => {
    if (guideDragIndex === null) {
      return
    }

    const handlePointerMove = (event: PointerEvent) => {
      if (!editorRef.current) {
        return
      }

      const bounds = editorRef.current.getBoundingClientRect()
      const point = {
        x: ((event.clientX - bounds.left) / bounds.width) * EDITOR_SIZE - EDITOR_CENTER,
        y: ((event.clientY - bounds.top) / bounds.height) * EDITOR_SIZE - EDITOR_CENTER,
      }

      if (!pointInPolygon(point, livePlanetPitchCurve) || distanceToOutline(point, livePlanetPitchCurve) < turningInset * 0.65) {
        return
      }

      setGuideControlPoints((current) =>
        (current.length === 4 ? current : activeGuideControlPoints).map((value, index) =>
          index === guideDragIndex ? point : value,
        ),
      )
    }

    const handlePointerUp = () => {
      setGuideDragIndex(null)
    }

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', handlePointerUp)
    return () => {
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', handlePointerUp)
    }
  }, [activeGuideControlPoints, guideDragIndex, livePlanetPitchCurve, turningInset])

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
    const sampleRingTeeth = DEFAULT_TOOTH_COUNT * 3
    const sampleSunTeeth = DEFAULT_TOOTH_COUNT * 2
    setControlRadii(GEARIFY_SAMPLE_RADII)
    setBaseBias(DEFAULT_BASE_BIAS)
    setSmoothing(DEFAULT_SMOOTHING)
    setToothCount(DEFAULT_TOOTH_COUNT)
    setRingToothCountInput(sampleRingTeeth)
    setSunToothCountInput(sampleSunTeeth)
    setToothDepth(DEFAULT_TOOTH_DEPTH)
    setToothSharpness(DEFAULT_TOOTH_SHARPNESS)
    setPlanetCount(3)
    setPlanetTurnsPerOrbit(6)
    setSunTurnsPerOrbit(3)
    setOrbitRadius(DEFAULT_ORBIT_RADIUS)
    setRingThickness(DEFAULT_RING_THICKNESS)
    setTurningInset(DEFAULT_TURNING_INSET)
    setSelectedTurningIndex(0)
    setBuildStage('diagram')
    setGuideControlPoints([])
    setAppliedInputs({
      controlRadii: [...GEARIFY_SAMPLE_RADII],
      baseBias: DEFAULT_BASE_BIAS,
      smoothing: DEFAULT_SMOOTHING,
      toothCount: DEFAULT_TOOTH_COUNT,
      ringToothCount: sampleRingTeeth,
      sunToothCount: sampleSunTeeth,
      toothDepth: DEFAULT_TOOTH_DEPTH,
      toothSharpness: DEFAULT_TOOTH_SHARPNESS,
      planetCount: 3,
      planetTurnsPerOrbit: 6,
      sunTurnsPerOrbit: 3,
      orbitRadius: DEFAULT_ORBIT_RADIUS,
      ringThickness: DEFAULT_RING_THICKNESS,
      turningInset: DEFAULT_TURNING_INSET,
      guideControlPoints: [],
    })
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
            <div className="card-title-actions">
              <span>{controlRadii.length} control points</span>
              <button className="ghost-button small" onClick={() => setGuideControlPoints(defaultGuideControlPoints)}>
                Reset guide
              </button>
            </div>
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
              d={pointsToSmoothPath(livePlanetPitchCurve, EDITOR_CENTER, EDITOR_CENTER)}
              className="pitch-curve"
            />
            <path
              d={pointsToSmoothPath(livePlanetOutline, EDITOR_CENTER, EDITOR_CENTER)}
              className="planet-outline"
            />
            {solvabilityGraph.path.length > 1 ? (
              <path
                d={pointsToOpenSmoothPath(solvabilityGraph.path, EDITOR_CENTER, EDITOR_CENTER)}
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
            {activeGuideControlPoints.length === 4 ? (
              <>
                <path
                  d={activeGuideControlPoints
                    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x + EDITOR_CENTER} ${point.y + EDITOR_CENTER}`)
                    .join(' ')}
                  className="guide-control-line"
                />
                {activeGuideControlPoints.map((point, index) => (
                  <circle
                    key={`guide-${index}`}
                    cx={point.x + EDITOR_CENTER}
                    cy={point.y + EDITOR_CENTER}
                    r={index === 0 || index === activeGuideControlPoints.length - 1 ? 6.5 : 5.2}
                    className="guide-handle"
                    onPointerDown={() => setGuideDragIndex(index)}
                  />
                ))}
              </>
            ) : null}
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
            Drag the white handles to shape the planet, then drag the green guide through the interior.
            The graph step samples that guide curve and only exposes intersections where the internal and
            external no-slip distances match.
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
            Ring tooth count
            <input
              type="number"
              min="6"
              max="240"
              value={ringToothCountInput}
              onChange={(event) => setRingToothCountInput(clamp(Number(event.target.value) || 6, 6, 240))}
            />
            <strong>{ringToothCountInput}</strong>
          </label>
          <label>
            Sun tooth count
            <input
              type="number"
              min="6"
              max="180"
              value={sunToothCountInput}
              onChange={(event) => setSunToothCountInput(clamp(Number(event.target.value) || 6, 6, 180))}
            />
            <strong>{sunToothCountInput}</strong>
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
          <div className="card-title-row">
            <button className="ghost-button small" onClick={applyRecalculation}>
              Recalculate
            </button>
            <span>{hasPendingSolveChanges ? 'Pending changes' : 'Up to date'}</span>
          </div>
          <div className="stage-toggle" role="tablist" aria-label="Generation steps">
            <button
              className={buildStage === 'diagram' ? 'stage-button active' : 'stage-button'}
              onClick={() => setBuildStage('diagram')}
            >
              1. Diagram
            </button>
            <button
              className={buildStage === 'ring' ? 'stage-button active' : 'stage-button'}
              onClick={() => setBuildStage('ring')}
            >
              2. Ring carve
            </button>
            <button
              className={buildStage === 'sun' ? 'stage-button active' : 'stage-button'}
              onClick={() => setBuildStage('sun')}
            >
              3. Sun carve
            </button>
            <button
              className={buildStage === 'assembly' ? 'stage-button active' : 'stage-button'}
              onClick={() => setBuildStage('assembly')}
            >
              4. Assembly
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
            <input
              type="number"
              min="0.25"
              max="24"
              step="0.25"
              value={planetTurnsPerOrbit}
              onChange={(event) => setPlanetTurnsPerOrbit(clamp(Number(event.target.value) || 0.25, 0.25, 24))}
            />
          </div>
          <div className="derived-metric">
            <span>Sun turns / orbit</span>
            <input
              type="number"
              min="0.25"
              max="24"
              step="0.25"
              value={sunTurnsPerOrbit}
              onChange={(event) => setSunTurnsPerOrbit(clamp(Number(event.target.value) || 0.25, 0.25, 24))}
            />
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
              min="0.002"
              max="0.2"
              step="0.002"
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
            <strong>{`${appliedInputs.toothCount} / ${sunToothCount} / ${ringToothCount}`}</strong>
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
              {buildStage === 'diagram'
                ? 'Step 1: solvability diagram'
                : buildStage === 'ring'
                  ? 'Step 2: ring carve'
                  : buildStage === 'sun'
                    ? 'Step 3: sun carve'
                    : 'Step 4: full assembly'}
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
          {buildStage === 'diagram' ? (
            <>
              <line x1="96" y1={VIEWBOX_SIZE - 92} x2={VIEWBOX_SIZE - 72} y2={VIEWBOX_SIZE - 92} className="diagram-axis" />
              <line x1="96" y1={VIEWBOX_SIZE - 92} x2="96" y2="84" className="diagram-axis" />
              <text x={VIEWBOX_SIZE - 62} y={VIEWBOX_SIZE - 76} className="diagram-axis-label">
                V
              </text>
              <text x="72" y="92" className="diagram-axis-label">
                D
              </text>
              <path d={pointsToOpenSmoothPath(diagramExternalCurve, 0, 0)} className="diagram-external" />
              <path d={pointsToOpenSmoothPath(diagramInternalCurve, 0, 0)} className="diagram-internal" />
              {diagramIntersections.map(({ candidate, point }, index) => (
                <g key={`diagram-intersection-${candidate.v}`}>
                  <circle
                    cx={point.x}
                    cy={point.y}
                    r={index === safeTurningIndex ? 10 : 7}
                    className={index === safeTurningIndex ? 'diagram-hit selected' : 'diagram-hit'}
                    onClick={() => setSelectedTurningIndex(index)}
                  />
                  <circle cx={point.x} cy={point.y} r="2.5" fill="#f8fff2" pointerEvents="none" />
                </g>
              ))}
              <text x="132" y="40" className="diagram-readout v">
                {`V = ${selectedTurningCenter.v?.toFixed(3) ?? '--'}`}
              </text>
              <text x="380" y="40" className="diagram-readout d">
                {`Distance = ${selectedTurningCenter.externalDistance?.toFixed(3) ?? '--'}`}
              </text>
              <text x="132" y="66" className="diagram-legend external">
                {`External Gear Graph Rotation Ratio: ${appliedInputs.planetTurnsPerOrbit.toFixed(2)}/1`}
              </text>
              <text x="132" y="90" className="diagram-legend internal">
                {`Internal Gear Graph Rotation Ratio: ${appliedInputs.sunTurnsPerOrbit.toFixed(2)}/1`}
              </text>
            </>
          ) : (
            <>
              <g transform={`translate(${CENTER} ${CENTER}) scale(${previewScale}) translate(${-CENTER} ${-CENTER})`}>
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
                    {(buildStage === 'ring' ? currentFrames.pitchFrames.slice(0, 1) : currentFrames.pitchFrames).map(
                      (planetPitch, index) => (
                        <path
                          key={`planet-pitch-${index}`}
                          d={pointsToSmoothPath(translatePath(planetPitch, { x: CENTER, y: CENTER }), 0, 0)}
                          fill="none"
                          stroke="#f2b14a"
                          strokeWidth="1.5"
                          opacity="0.42"
                        />
                      ),
                    )}
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
              </g>
              <text x="28" y="42" className="viewport-label">{`Pitch: ${toothPitch.toFixed(2)}`}</text>
              <text
                x="28"
                y="68"
                className="viewport-label"
              >{`Avg radii P/S: ${averagePlanetRadius.toFixed(1)} / ${averageSunRadius.toFixed(1)}`}</text>
              <text x="28" y="94" className="viewport-label">{`Loop turns: ${loopTurns}`}</text>
            </>
          )}
        </svg>
      </main>
    </div>
  )
}

export default App
