### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 81308e80-ed3e-4f1c-9293-b947735d555f
begin
	using CSV
	using DataFrames
	using Dates
	using Plots
	using PlutoUI
	using Serialization
end

# ╔═╡ 2914a464-ae24-4707-bb55-912454044e80
using StatsBase

# ╔═╡ 8900ad5c-ac19-11eb-2709-6bd4455db463
md"# Exploring ADNI data

This notebook will explore summary data provided by UC Berkeley and ADNI, based off sMRI and PET analysis conducted at UC Berkeley."

# ╔═╡ 9e980bfd-b5e9-4cef-bfda-cea9aa7c7897
md"## Loading the data as a DataFrame

First thing we need to do is to load in the data as a DataFrame so that it's easier for us to work with it."

# ╔═╡ 893ea036-110b-40de-828a-28f783cd5a18
root_dir = dirname(pwd())

# ╔═╡ 601d12de-c9cf-4ba5-893b-9730866af309
data_file = root_dir * "/data/UCBERKELEYAV1451_01_14_21.csv"

# ╔═╡ 565d7c0d-d60b-428b-9f83-bdbc20299f62
RIDDict = deserialize(root_dir * "/data/dicts/RIDDict")

# ╔═╡ 7c565aac-4aa5-4b8d-bed4-053a76a5aaea
TPIDDict = deserialize(root_dir * "/data/dicts/TPIDDict")

# ╔═╡ 45685366-fc07-4a05-a348-f709c7127327
data = DataFrame(CSV.File(data_file))

# ╔═╡ f67f9d4d-a783-4740-8a6d-ff2365bc8869
md"## Parcellation to Region map

To pick out data from particular brain regions, we'll need to make a dictionary that translates between feesurfer regions and the labels for those regionn in the data frame. This requires a little bit of formatting that it is included in the function `MakeParcellationDict`"

# ╔═╡ f2b4e579-e072-4016-99f3-eb03c0e1b0e0
function MakeParcellationDict(parcellation_file)
	names = Array{String}(undef, 83)
	
	parc = CSV.File(parcellation_file;delim=",")
	[names[i] = filter(x -> !isspace(x), parc[i][2]) for i in 1:83]
	[names[i] = replace(uppercase(names[i]), "-" => "_") for i in 1:length(names)]
	
	return Dict(zip(parc.index, names))
end

# ╔═╡ 651c664f-02bf-4646-bdee-4da1e9f35564
FSDict = MakeParcellationDict("/Users/pavanchaggar/Projects/NetworkTopology/data/sub-01_label-L2018_desc-scale1_stats.tsv")

# ╔═╡ 64247f09-1bee-4913-a2f8-8732fdd56d71
md" ## Accessing subject data
Subjects who have had multiple scanning sessions are repeated in the data. To isolate subject data, we'll collect the unique ID's into an array. For the purpose of model fitting, we may wish to focus on only subjects whohave multiple scans" 

# ╔═╡ 19bea830-004c-443e-9416-857d81d81e1e
subjectsID = unique(data.RID)

# ╔═╡ be12701c-943a-4ae7-94e4-48ad37e02f1a
subjectsManyNs = [i for (i,val) in countmap(data.RID) if val>2];

# ╔═╡ 91e29ec9-9326-49fc-88a4-478093fe2866
start_date = sort(data[:,:EXAMDATE])[1]

# ╔═╡ cb4bd167-5513-42dc-b1b3-23f5663b1977
end_date = sort(data[:,:EXAMDATE])[end]

# ╔═╡ 74542b84-a660-495c-b6fc-775de1bcaea4
md"subject = $(@bind subject Slider(1:1:108, show_value=true, default=10))"

# ╔═╡ 914a642e-dbac-4cce-ab33-cb2ae1c7f880
md"node = $(@bind node Slider(1:1:83, show_value=true, default=10))"

# ╔═╡ 9d13fa73-947c-4752-99f7-f9949f971ffa
subject_data = filter(x -> x.RID == subjectsManyNs[subject], data)

# ╔═╡ 445e7e94-e7a2-4ed9-9e57-e28b9bef4df3
scatter(subject_data[:,:EXAMDATE], subject_data[:,"BRAAK1_SUVR"], ylims=(0,3), xlims=(start_date, end_date))

# ╔═╡ Cell order:
# ╟─8900ad5c-ac19-11eb-2709-6bd4455db463
# ╠═81308e80-ed3e-4f1c-9293-b947735d555f
# ╟─9e980bfd-b5e9-4cef-bfda-cea9aa7c7897
# ╠═893ea036-110b-40de-828a-28f783cd5a18
# ╠═601d12de-c9cf-4ba5-893b-9730866af309
# ╠═565d7c0d-d60b-428b-9f83-bdbc20299f62
# ╠═7c565aac-4aa5-4b8d-bed4-053a76a5aaea
# ╠═45685366-fc07-4a05-a348-f709c7127327
# ╟─f67f9d4d-a783-4740-8a6d-ff2365bc8869
# ╠═f2b4e579-e072-4016-99f3-eb03c0e1b0e0
# ╠═651c664f-02bf-4646-bdee-4da1e9f35564
# ╟─64247f09-1bee-4913-a2f8-8732fdd56d71
# ╠═19bea830-004c-443e-9416-857d81d81e1e
# ╠═2914a464-ae24-4707-bb55-912454044e80
# ╠═be12701c-943a-4ae7-94e4-48ad37e02f1a
# ╠═91e29ec9-9326-49fc-88a4-478093fe2866
# ╠═cb4bd167-5513-42dc-b1b3-23f5663b1977
# ╟─74542b84-a660-495c-b6fc-775de1bcaea4
# ╟─914a642e-dbac-4cce-ab33-cb2ae1c7f880
# ╠═445e7e94-e7a2-4ed9-9e57-e28b9bef4df3
# ╠═9d13fa73-947c-4752-99f7-f9949f971ffa
