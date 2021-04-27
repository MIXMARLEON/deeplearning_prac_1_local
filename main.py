import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tkinter.filedialog import *
from tensorflow import keras
import PIL.ImageGrab as ImageGrab
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

root = Tk()
root.title("MNIST form")
#  These variables store the state of the radio buttons.
r_var = BooleanVar()
r_var.set(0)


#  Function convert images array format
def optimize_img(img):
    img = keras.preprocessing.image.img_to_array(img)
    img = img.reshape(28, 28)
    img = img.astype('float32')
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)
    return img


def prediction(img):
    """Function returns array of predictions"""
    model = keras.models.load_model('C:/deeplearning/models/cnn_mnist2.h5')
    return model.predict(img)


def plot_draw_result(prediction, true_label, img, root):
    class_names = ['0', '1', '2', '3', '4',
                   '5', '6', '7', '8', '9']
    fig = Figure(figsize=(6, 3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(img, cmap='binary')
    predicted_label = np.argmax(prediction)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    ax1.set_xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(prediction),
                                             true_label),
                   color=color)

    ax2.set_xticks(range(10))
    ax2.set_yticks([])
    thisplot = ax2.bar(range(10), prediction[0], color="#777777")
    ax2.set_ylim([0, 1])
    predicted_label = np.argmax(prediction)

    thisplot[predicted_label].set_color('red')
    thisplot[int(true_label)].set_color('blue')

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

        #  Placement of elements
        self.entry.grid(row=0, column=0, sticky=W+E)
        self.open_button.grid(row=0, column=1, sticky=W+E)
        self.radio_handwrite.grid(row=0, column=2, sticky=W+E)
        self.radio_link.grid(row=0, column=3, sticky=W+E)
        self.paint_canvas.grid(row=1, column=0)

        self.clear_button.grid(row=2, column=0, sticky=W+E)
        self.process_button.grid(row=2, column=1, columnspan=2, sticky=W+E)
        self.predict_lable.grid(row=4, column=0, sticky=W+E)
        self.predict_entry.grid(row=4, column=1, columnspan=2, sticky=W+E)

    #  This function allows to paint on paint_canvas
    def paint(self, event):
        brush_size = 10
        x1 = event.x + brush_size
        x2 = event.x - brush_size
        y1 = event.y + brush_size
        y2 = event.y - brush_size
        self.paint_canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")

    def open_img(self, event):
        file = askopenfilenames()
        self.entry.insert(END, file)

    def clear_paint(self, event):
        self.paint_canvas.delete("all")

    def process(self, event):
        img, img_arr = None, None
        if r_var.get():
            img = self.entry.get()
            img = keras.preprocessing.image.load_img(img, color_mode='grayscale', target_size=(28, 28))
            img_arr = optimize_img(img)

        #  If handwrite radiobutton active
        if not r_var.get():
            x = root.winfo_rootx() + self.paint_canvas.winfo_x()
            y = root.winfo_rooty() + self.paint_canvas.winfo_y()
            x1 = x + self.paint_canvas.winfo_width()
            y1 = y + self.paint_canvas.winfo_height()
            #  Calculates the location of the window and canvas, crops the drawn image
            img = ImageGrab.grab().crop((x, y, x1, y1))
            img = img.resize((28, 28), ImageGrab.Image.ANTIALIAS)
            #  Convert image in grayscale palette
            img = img.convert('L')

        img_arr = optimize_img(img)
        pred = prediction(img_arr)
        pred_lable = self.predict_entry.get()
        fig = plot_draw_result(pred, pred_lable, img, root)
        self.result_canvas = FigureCanvasTkAgg(fig, master=root)
        self.result_canvas.draw()
        self.result_canvas.get_tk_widget().grid(row=1, column=1, columnspan=3)
        self.entry.delete(0, END)

form = MyForm(root)
root.mainloop()