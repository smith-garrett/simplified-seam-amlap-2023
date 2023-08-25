# Basic functions for setting up ToySWIFT models

import Plots: plot
using DataFrames
using CSV
using Distributions


"""
	proc_rates(nwords, word, ν)

Return the processing rates λ(w, t) for word w at time t given the currently fixated word and the activation spreading parameter ν.
"""
function proc_rates(nwords, currword, ν)
	@assert 0 < currword ≤ nwords "Current word must fall between 1 and nwords"
	
	# Calculate the normalization constant
	σ = 1 / (1 + 2*ν + ν^2)

	# Setting the processing rates
	#λ = zeros(nwords)
    # This could cause problems with auto.diff.
	λ = zeros(eltype(ν), nwords)
	(currword - 1) ≥ 1 && (λ[currword - 1] = σ * ν)
	λ[currword] = σ
	(currword + 1) ≤ nwords && (λ[currword + 1] = σ * ν)
	(currword + 2) ≤ nwords && (λ[currword + 2] = σ * ν^2)

	return λ
end


"""
    update_activations!(activations, λ, duration, r, maxes=ones(length(activations)))

Returns the updated activations of all words, modified in-place. Bounds activations at maxes.
"""
function update_activations!(activations, λ, duration, r, maxes=ones(length(activations)))
	# Update activations on the second scale, ∴ / 1000
	activations .= min.(activations .+ r .* λ .* duration ./ 1000, maxes)
	return activations
end


"""
    update_activations(activations, λ, duration, r, maxes)

Returns the updated activations of all words. Bounds activations at maxes.
"""
function update_activations(activations, λ, duration, r, maxes=ones(length(activations)))
	# Update activations on the second scale, ∴ / 1000
	return min.(activations .+ r .* λ .* duration ./ 1000, maxes)
end


"""
	update_saliencies!(saliencies, activations, η=-3.0, 						
					  maxes=ones(length(activations)))

Update the saliencies for each word in-place based on the vector activations and the small baseline saliency 10^η.
"""
function update_saliencies!(saliencies, activations, η=-3.0, maxes=ones(length(activations)))
	saliencies .= maxes .* sinpi.(activations ./ maxes) .+ exp10(η)
	return saliencies
end


"""
	update_saliencies(saliencies, activations, η=-3.0, 						
					  maxes=ones(length(activations)))

Update the saliencies for each word based on the vector activations and the small baseline saliency 10^η.
"""
function update_saliencies(saliencies, activations, η=-3.0, maxes=ones(length(activations)))
	return maxes .* sinpi.(activations ./ maxes) .+ exp10(η)
end


"""
	update_probs!(probs, saliencies, γ=1.0)

Update probabilities in-place using the noise parameter γ.
"""
function update_probs!(probs, saliencies, γ=1.0)
	probs .= (saliencies.^γ) ./ sum(saliencies.^γ)
end


"""
	update_probs(probs, saliencies, γ=1.0)

Update probabilities using the noise parameter γ.
"""
function update_probs(probs, saliencies, γ=1.0)
	return (saliencies.^γ) ./ sum(saliencies.^γ)
end


"""
	get_left_saliency(saliencies, currword, κ=2.0)

Get the product (!) of saliencies of words to the left of the current word.
"""
function get_left_saliency(saliencies, currword, κ=2.0)
	#return currword > 1 ? prod(1 .+ κ .* saliencies[1:(currword-1)]) : one(eltype(saliencies))
	return currword > 1 ? prod(saliencies[1:(currword-1)]) : one(eltype(saliencies))
	# Also need to subtract 10^-eta for complete correctness
end


"""
	get_left_activation(activations, currword, maxactivations)

Get the product (!) of saliencies of words to the left of the current word.
"""
function get_left_activation(activations, currword, maxactivations)
	return currword > 1 ? sum(maxactivations[1:(currword-1)] .- activations[1:(currword-1)]) : zero(eltype(activations))
end


abstract type AbstractScanpathResult end


"""
    SmoothScanpathResult

A structure containing a scanpath, smooth saliency trajectories, and other information about
a ToySWIFT-parsed sentence.
"""
struct SmoothScanpathResult <: AbstractScanpathResult
    scanpath::DataFrame
    saliencies::AbstractArray
    nwords::Int
    words
end


"""
    DiscreteScanpathResult

A structure containing a scanpath, saliency trajectories saved at the discrete time points
when saccades are made, and other information about a ToySWIFT-parsed sentence.
"""
struct DiscreteScanpathResult <: AbstractScanpathResult
    scanpath::DataFrame
    saliencies::AbstractArray
    nwords::Int
    words
end


"""
	generate_scanpath(logfreq, maxlogfreq;
					  ν=0.3, r=10, η=-3.0, μ=200, ι=0.5, β=0.6, γ=1.0, ζ=0.75)

Generate a random scanpath for a sentence with as many words as there are log frequencies in
logfreq using the parameters provided. Incorporates parsing via the argument deps, which
should be a list of lists of word indices.

Parsing can be turned off by passing ζ = nothing.
"""
function generate_scanpath(logfreq,
    maxlogfreq,
    deps,
    words;
    ν=0.3,
    r=10,
    η=-3.0,
    μ=200,
    ι=0.5,
    β=0.6,
    γ=1.0,
    ζ=0.75,
)
	nwords = length(logfreq)
	shape = 9
	rate = shape / μ
	maxactivations = 1 .- β .* logfreq ./ maxlogfreq
	activations = zeros(nwords)
	saliencies = zeros(nwords)
	time = 0.0
	currdur = 0.0
	currword = 1
	probs = zeros(nwords)
	#probs[1] = 1.0
	scanpath = []
	salhist = zeros((1, nwords))
    not_yet_parsed = fill(true, nwords)

	while any(activations .< maxactivations) && currword < nwords
		# Calculate current fixation duration
		#leftsal = get_left_saliency(saliencies, currword, κ)
		leftsal = get_left_activation(activations, currword, maxactivations) / currword
		#currrate = rate * (1 + ι * activations[currword]) / (1 + κ * leftsal)
		currrate = rate * exp(ι * (activations[currword] - leftsal))
		#currrate = rate + ι * activations[currword] - κ * leftsal
		#currrate = rate * (1 + ι * activations[currword] - κ * leftsal)
		currdur = rand(Gamma(shape, currrate^-1))

		# Save and update
		push!(scanpath, (word = currword, time = time, duration = currdur))
		time += currdur
		
		# Get processing rates
		λ = proc_rates(nwords, currword, ν)
		
		# Update activations
		update_activations!(activations, λ, currdur, r, maxactivations)

        # Parsing: whenever a word reaches max activation, reactivate its dependencies
        if !isnothing(ζ)
            idx = findall(activations .>= maxactivations)
            for i in idx
                if not_yet_parsed[i] && !isempty(deps[i])
                    activations[deps[i]] .= ζ .* maxactivations[deps[i]]
                end
            end
            not_yet_parsed[idx] .= false
        end

		# Update saliencies
		update_saliencies!(saliencies, activations, η, maxactivations)

		# Saving saliencies
		salhist = vcat(salhist, saliencies')

		# Saccade target selection
		update_probs!(probs, saliencies, γ)
		currword = rand(Categorical(probs))
	end
	# Save last saccade
	push!(scanpath, (word = currword, time = time, duration = currdur))
	#return DataFrame(scanpath), salhist
    return DiscreteScanpathResult(DataFrame(scanpath), salhist, nwords, words)
end


"""
	generate_smooth_scanpath(logfreq, maxlogfreq, deps, words;
					  ν=0.3, r=10, η=-3.0, μ=200, ι=0.5, β=0.6, γ=1.0, ζ=0.75)

Generate a random scanpath for a sentence with as many words as there are log frequencies in
logfreq using the parameters provided. Incorporates parsing via the argument deps, which
should be a list of lists of word indices.

Parsing can be turned off by passing ζ = nothing.
"""
function generate_smooth_scanpath(logfreq,
    maxlogfreq,
    deps,
    words;
    ν=0.3,
    r=10,
    η=-3.0,
    μ=200,
    ι=0.5,
    β=0.6,
    γ=1.0,
    ζ=0.75,
)	
    nwords = length(logfreq)
	shape = 9
	rate = shape / μ
	maxactivations = 1 .- β .* logfreq ./ maxlogfreq
	activations = zeros(nwords)
	saliencies = zeros(nwords)
	time = 0.0
	currdur = 0.0
	currword = 1
	probs = zeros(nwords)
	#probs[1] = 1.0
	scanpath = []
	salhist = zeros((1, nwords))
    not_yet_parsed = fill(true, nwords)
    activated = fill(false, nwords)

	while any(activations .< maxactivations) && currword < nwords
		# Calculate current fixation duration
		#leftsal = get_left_saliency(saliencies, currword, κ)
		leftact = get_left_activation(activations, currword, maxactivations) / currword
		#currrate = rate * (1 + ι * activations[currword]) / (1 + κ * leftsal)
		currrate = rate * exp(ι * (activations[currword] - leftact))
		currdur = rand(Gamma(shape, currrate^-1))

		# Save and update
		push!(scanpath, (word = currword, time = time, duration = currdur))
		time += currdur
		
		# Get processing rates
		λ = proc_rates(nwords, currword, ν)
		
		for t in 1:currdur
			# Update activations
			update_activations!(activations, λ, 1.0, r, maxactivations)

			# Parsing: whenever a word reaches max activation, reactivate its dependencies
            if !isnothing(ζ)
                idx = findall(activations .>= maxactivations)
                for i in idx
                    if not_yet_parsed[i] && !isempty(deps[i])
                        activations[deps[i]] .= ζ .* maxactivations[deps[i]]
                    end
                end
                not_yet_parsed[idx] .= false
            end

			# Update saliencies
			update_saliencies!(saliencies, activations, η, maxactivations)

			# Saving saliencies
			salhist = vcat(salhist, saliencies')
		end

		# Saccade target selection
		update_probs!(probs, saliencies, γ)
		currword = rand(Categorical(probs))
	end
	# Save last saccade
	push!(scanpath, (word = currword, time = time, duration = currdur))
	#return DataFrame(scanpath), salhist
    return SmoothScanpathResult(DataFrame(scanpath), salhist, nwords, words)
end


"""
    str_to_vec(x) 

Convert a string in the format "[1, 2]" to a vector.
"""
function str_to_vec(x)
	if x == "[]"
		return Int[]
	else
		return parse.(Int, split(strip(x, ['[', ']']), ", "))
	end
end
#str_to_vec(x) = [parse(Int, t.match) for t in eachmatch(r"([0-9])", x)]


"""
	generate_data(df)

Generate ToySWIFT scanpaths from a corpus in df. Assumes the sentence/item number is in the
column sentID. Currently uses the default parameter values of generate_scanpath(...).
"""
function generate_data(df)
	data = DataFrame()
	for item in unique(df.sentID)
		logfreqs = df[df.sentID .== item, :logfreq]
        deps = str_to_vec.(df[df.sentID .== item, :deps])
		sp, _ = generate_scanpath(logfreqs, maximum(df.logfreq), deps)
		sp.item .= item
		sp.fixation_number = 1:nrow(sp)
		sp.logfreq = logfreqs[sp.word]
		append!(data, sp)
	end
	return data
end


"""
    plot(x::SmoothScanpathResult)

Plot a scanpath with smooth saliency trajectories.
"""
function Plots.plot(x::SmoothScanpathResult)
	plotsal = x.saliencies .+ reshape(collect(1:x.nwords), (1, x.nwords))
	plot(plotsal, 1:size(plotsal, 1), legend=nothing, xmirror=true)
	plot!(x.scanpath.word, x.scanpath.time, linetype=:steppre, color=:black, linewidth=2)
	yflip!()
	xticks!(1:x.nwords, x.words)
	ylabel!("Time (ms)")
end


"""
    plot(x::DiscreteScanpathResult)

Plot a scanpath with discrete saliency trajectories.
"""
function Plots.plot(x::DiscreteScanpathResult)
	plotsal = x.saliencies .+ reshape(collect(1:x.nwords), (1, x.nwords))
	plot(plotsal, x.scanpath.time, legend=nothing, xmirror=true)
	plot!(x.scanpath.word, x.scanpath.time, linetype=:steppre, color=:black, linewidth=2)
	yflip!()
	xticks!(1:x.nwords, x.words)
	ylabel!("Time (ms)")
end

