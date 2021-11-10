import os
import numpy as np
import PIL.ImageGrab as ImageGrab
from tkinter.filedialog import *
from tensorflow import keras
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


"""

This program is designed to test a convolutional neural network
based on VGG16.It allows you to enter a handwritten number or
open a file on your computer. After processing, the result
is displayed in the form of a processed image and a bar chart,
which shows the degree of "confidence" of the network
that the image corresponds to each of the 10 digits.

"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
root = Tk()  # Creating the window of program.
root.title("MNIST form")

#  These variables store the state of the radio buttons.
r_var = BooleanVar()
r_var.set(0)


def optimize_img(img):
    """Function convert images to array format"""
    img = keras.preprocessing.image.img_to_array(img)
    img = img.reshape(28, 28)
    img = img.astype('float32')
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)
    return img


def predict(img):
    """Function returns array of predictions

    Array of 10 numbers.
    They represent the "confidence" of the network that the
    image corresponds to each of the 10 different numbers.

    """
    model = keras.models.load_model('models/cnn_mnist_vgg16.h5')
    return model.predict(img)


def plot_draw_result(prediction, true_label, img, root):
    """Function to drawing result figure.

    The function creates two subplots, one representing user's drawing, the other representing bar graph.
    Under the user's image creating label with the predicted result and the neural network confidence in percent.
    If answer is correct - color of label turns blue, if  answer is not correct - red.
    Columns represent each digit from 0 to 9.
    Height of column represents degree of confidence network has in range from 0 to 1.
    If on graph answer is correct, the correct column turns blue.
    If answer is incorrect, predicted value column turns red and correct value column turns blue.


    :param prediction: Array with predictions.

    :param true_label: String with true digit label.
    :param img: PIL img.
    :param root: TKinter active root.
    :return: Figure with two subplots.


    """

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    fig = Figure(figsize=(6, 3))  # base to subplots
    ax1 = fig.add_subplot(121)   # subplot to image
    ax2 = fig.add_subplot(122)  # subplot to bar graph

    # Removing labels of image axes and showing it in grayscale palette.
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(img, cmap='binary')

    predicted_label = np.argmax(prediction)  # the highest value matches the predicted value
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    ax1.set_xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(prediction),
                                             true_label),
                   color=color)

    #  creating bar graph
    ax2.set_xticks(range(10))
    this_plot = ax2.bar(range(10), prediction[0], color="#777777")
    ax2.set_ylim([0, 1])
    predicted_label = np.argmax(prediction)
    this_plot[predicted_label].set_color('red')
    this_plot[int(true_label)].set_color('blue')

    return fig


class MyForm:
    def __init__(self, root):
        global r_var

        #  Frames
        self.top_frame = Frame(root)
        self.center_frame = Frame(root)
        self.bottom_frame = Frame(root)

        #  Top elements
        self.entry = Entry(root)
        self.open_button = Button(root, text="Open")
        self.radio_handwrite = Radiobutton(root, text="Handwrite", variable=r_var, value=0)
        self.radio_link = Radiobutton(root, text="Link", variable=r_var, value=1)

        #  Center elements
        self.paint_canvas = Canvas(root, height=280, width=280, bg='white')
        self.result_canvas = Canvas(root)
        self.clear_button = Button(root, text="Clear")

        #  Bottom elements
        self.process_button = Button(root, text="Process")
        self.predict_entry = Entry(root)
        self.predict_lable = Label(root, text="Enter the true label of your number:")

        #  Binds
        self.open_button.bind("<Button-1>", self.open_img)
        self.clear_button.bind("<Button-1>", self.clear_paint)
        self.process_button.bind("<Button-1>", self.process)
        self.paint_canvas.bind("<B1-Motion>", self.paint)

        # Placement of top elements
        self.entry.grid(row=0, column=0, sticky=W+E)
        self.open_button.grid(row=0, column=1, sticky=W+E)
        self.radio_handwrite.grid(row=0, column=2, sticky=W+E)
        self.radio_link.grid(row=0, column=3, sticky=W+E)

        # Canvas with results
        self.paint_canvas.grid(row=1, column=0)

        # Placement of bottom elements
        self.clear_button.grid(row=2, column=0, sticky=W+E)
        self.process_button.grid(row=2, column=1, columnspan=2, sticky=W+E)
        self.predict_lable.grid(row=4, column=0, sticky=W+E)
        self.predict_entry.grid(row=4, column=1, columnspan=2, sticky=W+E)

    #  This function allows to paint on paint_canvas
    def paint(self, event):
        """Drawing while user hold left mouse button."""
        brush_size = 10  # Brush size in pixels
        x1 = event.x + brush_size
        x2 = event.x - brush_size
        y1 = event.y + brush_size
        y2 = event.y - brush_size
        self.paint_canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")

    def open_img(self, event):
        """Inserts link to file in the end of entry field."""
        file = askopenfilenames()
        self.entry.insert(END, file)

    def clear_paint(self, event):
        """Cleans up canvas for handwriting input."""
        self.paint_canvas.delete("all")

    def process(self, event):
        #  If "link" radiobutton active.
        if r_var.get():
            img = self.entry.get()
            img = keras.preprocessing.image.load_img(img, color_mode='grayscale', target_size=(28, 28))

        #  If "Handwrite" radiobutton active.
        if not r_var.get():

            #  Calculates the location of the window and canvas, crops the drawn image.
            x = root.winfo_rootx() + self.paint_canvas.winfo_x()
            y = root.winfo_rooty() + self.paint_canvas.winfo_y()
            x1 = x + self.paint_canvas.winfo_width()
            y1 = y + self.paint_canvas.winfo_height()
            img = ImageGrab.grab().crop((x, y, x1, y1))

            img = img.resize((28, 28), ImageGrab.Image.ANTIALIAS)

            #  Convert image in grayscale palette.
            img = img.convert('L')

        img_arr = optimize_img(img)  # image array
        prediction = predict(img_arr)  # array with predictions
        true_lable = self.predict_entry.get()  # true label which user entered
        fig = plot_draw_result(prediction, true_lable, img, root)

        # Drawing result canvas.
        self.result_canvas = FigureCanvasTkAgg(fig, master=root)
        self.result_canvas.draw()
        self.result_canvas.get_tk_widget().grid(row=1, column=1, columnspan=3)

        # Clearing the link input field.
        self.entry.delete(0, END)


form = MyForm(root)
root.mainloop()