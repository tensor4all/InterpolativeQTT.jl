"""
    Interval{V}

A closed-open interval `[start, stop)` over values of type `V`. The constructor enforces
`start ≤ stop` by sorting the arguments.
"""
struct Interval{V}
    start::V
    stop::V

    function Interval{V}(start::V, stop::V) where {V}
        return new{V}(min(start, stop), max(start, stop))
    end
end

function Base.in(v::V, interval::Interval{V}) where {V}
    return v >= interval.start && v < interval.stop
end

"""
    midpoint(interval::Interval{V}) -> V

Return the midpoint of `interval`.
"""
function midpoint(interval::Interval{V}) where {V}
    return (interval.start + interval.stop) / 2
end

"""
    split(interval::Interval{V}, value::V = midpoint(interval)) -> Vector{Interval{V}}

Split `interval` at `value` (defaulting to its midpoint). Returns two sub-intervals if
`value` lies inside `interval`, or the original interval unchanged otherwise.
"""
function split(interval::Interval{V}, value::V = midpoint(interval)) where {V}
    if value in interval
        return [Interval{V}(interval.start, value), Interval{V}(value, interval.stop)]
    else
        return [interval]
    end
end

"""
    intervallength(interval::Interval{V}) -> V

Return the length `stop - start` of `interval`.
"""
function intervallength(interval::Interval{V}) where {V}
    return interval.stop - interval.start
end

"""
    NInterval{N, V}

An `N`-dimensional axis-aligned box `[start, stop)` with corner coordinates stored as
`NTuple{N, V}`. The constructor enforces `start[i] ≤ stop[i]` component-wise.
"""
struct NInterval{N, V}
    start::NTuple{N, V}
    stop::NTuple{N, V}

    function NInterval{N, V}(start::NTuple{N, V}, stop::NTuple{N, V}) where {N, V}
        return new{N, V}(tuple(min.(start, stop)...), tuple(max.(start, stop)...))
    end
end

function Base.in(v::NTuple{N, V}, interval::NInterval{N, V}) where {N, V}
    return all(v .>= interval.start) && all(v .< interval.stop)
end

#function midpoint(interval::NInterval{N,V})::NTuple{N,V} where {N, V}
#return tuple(((interval.start .+ interval.stop) ./ 2)...)
#end

"""
    split(interval::NInterval{N, V}) -> Vector{NInterval{N, V}}

Bisect `interval` along every dimension, returning the `2^N` equal sub-boxes.
"""
function split(interval::NInterval{N, V}) where {N, V}
    intervals = NInterval{N, V}[]
    halfintervallength = intervallength(interval) ./ 2
    for shifts in Iterators.product(ntuple(x -> 0:1, N)...)
        start_ = interval.start .+ shifts .* halfintervallength
        push!(intervals, NInterval{N, V}(start_, start_ .+ halfintervallength))
    end
    return intervals
end

"""
    intervallength(interval::NInterval{N, V}) -> NTuple{N, V}

Return the side lengths `stop .- start` of the N-dimensional box `interval`.
"""
function intervallength(interval::NInterval{N, V})::NTuple{N, V} where {N, V}
    return tuple((interval.stop .- interval.start)...)
end
