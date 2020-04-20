"""
Original code by Adrian Rosebrock on March 23, 2015
https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
"""


def sliding_window(image, step_size, window_size):
    # slide a window across the image
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            # yield the current window
            yield x, y, image[y:y + window_size[1], x:x + window_size[0]]


"""
# Example implementation
for (x, y, window) in sliding_window(image, step_size=X, window_size=(windowW, windowH)):
    # if the window does not meet our desired window size, ignore it
    if window.shape[0] != winH or window.shape[1] != winW:
        continue
        
    # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
    # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
    # WINDOW
    # since we do not have a classifier, we'll just draw the window
    # plt.imshow(window)
    # plt.show()
    
    clone = cropped_image.copy()
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    cv2.imshow("Window", clone)
    cv2.waitKey(1)
    time.sleep(0.025)
"""
