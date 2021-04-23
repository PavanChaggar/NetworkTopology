### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ 70d11120-9d4d-11eb-14a1-2373f9ca666a
struct MyStruct
	a
	b
	MyStruct(a,b) = new(b, a)
end

# ╔═╡ b198a09f-c953-4947-8a76-23bc3cf81fd4
A = MyStruct(1.0, 2.0)

# ╔═╡ 513f1f50-875b-42a6-8aee-d8a21cba776e


# ╔═╡ Cell order:
# ╠═70d11120-9d4d-11eb-14a1-2373f9ca666a
# ╠═b198a09f-c953-4947-8a76-23bc3cf81fd4
# ╠═513f1f50-875b-42a6-8aee-d8a21cba776e
