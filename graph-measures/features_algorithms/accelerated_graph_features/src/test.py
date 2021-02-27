import _features as af
af.test()
nodes = af.example_feature({
	"indices":[0, 3, 3, 4, 6],
	"neighbors": [1, 2, 3, 0, 1, 2]
})

print(nodes)
