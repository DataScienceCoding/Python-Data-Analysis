import scipy.misc
import matplotlib.pyplot as plt

face = scipy.misc.face()
print(face.shape, ':face shape\n')
xmax = face.shape[0]
ymax = face.shape[1]
face = face[:min(xmax, ymax), :min(xmax, ymax)].copy()
xmax = face.shape[0]
ymax = face.shape[1]
face[range(xmax), range(ymax)] = 255
face[range(xmax-1, -1, -1), range(ymax)] = 255

print(face)

plt.imshow(face)
plt.show()

# acopy = face.copy()
# aview = face.view()
# aview.flat = 0

# plt.subplot(221)
# plt.imshow(face)
# plt.subplot(222)
# plt.imshow(acopy)
# plt.subplot(223)
# plt.imshow(aview)
# plt.show()
