using Test

testcases = filter(f -> endswith(f, ".jl") & (f != "runtests.jl"), readdir())

for testcase in testcases
    @testset "$(testcase)" begin
        include(joinpath(testcase))
    end
end
