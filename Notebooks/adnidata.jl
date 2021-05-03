### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 81308e80-ed3e-4f1c-9293-b947735d555f
using CSV

# ╔═╡ 8900ad5c-ac19-11eb-2709-6bd4455db463
md"# Exploring ADNI data

This notebook will explore summary data provided by UC Berkeley and ADNI, based off sMRI and PET analysis conducted at UC Berkeley."

# ╔═╡ 601d12de-c9cf-4ba5-893b-9730866af309
data_file = "/Users/pavanchaggar/Projects/NetworkTopology/data/UCBERKELEYAV1451_01_14_21.csv"

# ╔═╡ 45685366-fc07-4a05-a348-f709c7127327
data = DataFrame(CSV.File(data_file))

# ╔═╡ 43d4ae76-30d8-4311-8e8e-74dc6892d107
filter(x -> x.RID == 56, data)

# ╔═╡ 4b8d985b-a312-4d00-b039-c594dceb9983
parcellation_file = "/Users/pavanchaggar/Projects/NetworkTopology/data/sub-01_label-L2018_desc-scale1_stats.tsv"

# ╔═╡ 3e9f4a7f-0ad9-4611-893a-de818a8973f3
parc = CSV.File(parcellation_file;delim=",");

# ╔═╡ ae87abdd-04ef-4475-a554-e2b6d7737102
parcarray = Array(parc);

# ╔═╡ e958982b-dbd0-45b3-901d-fe6a5ccc52c9
names = Array{String}(undef, 83)

# ╔═╡ 8038d132-4ca9-43a8-9c52-6690e04218e0
[names[i] = filter(x -> !isspace(x), parcarray[i][2]) for i in 1:83]

# ╔═╡ 6856d5e6-6569-4b6a-8f68-f0e6b798ba8d
[names[i] = replace(uppercase(names[i]), "-" => "_") for i in 1:length(names)]

# ╔═╡ c3f14899-bdcb-453c-b84f-9dea44c5653a
freesurfer_dict = Dict(zip(collect(1:83), names))

# ╔═╡ b9261b6b-0521-40d4-be43-04bd9961e65d
filter(x -> x.RID == 56, data)[:,freesurfer_dict[1]*"_SUVR"]

# ╔═╡ Cell order:
# ╟─8900ad5c-ac19-11eb-2709-6bd4455db463
# ╠═81308e80-ed3e-4f1c-9293-b947735d555f
# ╠═601d12de-c9cf-4ba5-893b-9730866af309
# ╠═45685366-fc07-4a05-a348-f709c7127327
# ╠═43d4ae76-30d8-4311-8e8e-74dc6892d107
# ╠═4b8d985b-a312-4d00-b039-c594dceb9983
# ╠═3e9f4a7f-0ad9-4611-893a-de818a8973f3
# ╠═ae87abdd-04ef-4475-a554-e2b6d7737102
# ╠═e958982b-dbd0-45b3-901d-fe6a5ccc52c9
# ╠═8038d132-4ca9-43a8-9c52-6690e04218e0
# ╠═6856d5e6-6569-4b6a-8f68-f0e6b798ba8d
# ╠═c3f14899-bdcb-453c-b84f-9dea44c5653a
# ╠═b9261b6b-0521-40d4-be43-04bd9961e65d
