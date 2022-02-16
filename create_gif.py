import imageio
images = []
filenames = [f"iters/iter{i}.png" for i in range(2, 99)]
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('progress.gif', images)