import sys
from PyQt5.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QMainWindow, QTextEdit, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton
from PyQt5.QtCore import Qt
import re
from data.token_completion import preprocess as pp
import generator

class InputTextEdit(QTextEdit):
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Tab:
            # Aggiunge due spazi invece di inserire il carattere di 'tab'
            cursor = self.textCursor()
            cursor.insertText("  ")
        else:
            super().keyPressEvent(event)

class MainWindow(QMainWindow):
    
    def initUI(self):
        self.setGeometry(100, 100, 1000, 600) # set the window size to 640x480 pixels
        self.setWindowTitle('nanoGPTxCodeCompletion')
        self.show()

    def __init__(self):
        super().__init__()
        self.initUI()

        self.generator = generator.Generator()

        # Crea la text box per l'input
        self.input_textbox = InputTextEdit()
        # self.input_textbox.textChanged.connect(self.generate_text)

        # Crea la text box per l'output
        self.output_textbox = QTextEdit()
        self.output_textbox.setReadOnly(True)
        
        # Crea i selettori di parametri
        model_box_label = QLabel("Select model")
        self.model_box = QComboBox(self)
        self.model_box.addItems(["nanoGPTxCodeCompletion"])
        self.model_box.setCurrentText("nanoGPTxCodeCompletion")
        self.model_box.currentIndexChanged.connect(self.change_model)

        self.num_samples_spinbox = QSpinBox()
        self.num_samples_spinbox.setValue(self.generator.num_samples)
        self.num_samples_spinbox.setMinimum(1)
        self.num_samples_spinbox.setMaximum(10)
        self.num_samples_spinbox.valueChanged.connect(self.set_num_samples)

        self.max_new_tokens_spinbox = QSpinBox()
        self.max_new_tokens_spinbox.setValue(self.generator.max_new_tokens)
        self.max_new_tokens_spinbox.setMinimum(1)
        self.max_new_tokens_spinbox.setMaximum(1000)
        self.max_new_tokens_spinbox.valueChanged.connect(self.set_max_new_tokens)

        self.temperature_spinbox = QDoubleSpinBox()
        self.temperature_spinbox.setValue(self.generator.temperature)
        self.temperature_spinbox.setMinimum(0)
        self.temperature_spinbox.setSingleStep(0.01)
        self.temperature_spinbox.setMaximum(1)
        self.temperature_spinbox.valueChanged.connect(self.set_temperature)

        self.top_k_spinbox = QSpinBox()
        if self.generator.top_k is not None:
            self.top_k_spinbox.setValue(self.generator.top_k)
        else:
            self.top_k_spinbox.setValue(0)
        self.top_k_spinbox.setMinimum(0)
        self.top_k_spinbox.setMaximum(1000)
        self.top_k_spinbox.valueChanged.connect(self.set_top_k)

        # Crea i label per i selettori di parametri
        num_samples_label = QLabel("Num Samples:")
        max_new_tokens_label = QLabel("Max New Tokens:")
        temperature_label = QLabel("Temperature:")
        top_k_label = QLabel("Top K:")

        # Crea il bottone "Generate"
        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.generate_text)
        # Crea il bottone "Clear"
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_text)

        # Crea il layout della sidebar
        sidebar_layout = QVBoxLayout()
        sidebar_layout.addWidget(model_box_label)
        sidebar_layout.addWidget(self.model_box)
        sidebar_layout.addWidget(num_samples_label)
        sidebar_layout.addWidget(self.num_samples_spinbox)
        sidebar_layout.addWidget(max_new_tokens_label)
        sidebar_layout.addWidget(self.max_new_tokens_spinbox)
        sidebar_layout.addWidget(temperature_label)
        sidebar_layout.addWidget(self.temperature_spinbox)
        sidebar_layout.addWidget(top_k_label)
        sidebar_layout.addWidget(self.top_k_spinbox)
        sidebar_layout.addWidget(self.generate_button)
        sidebar_layout.addWidget(self.clear_button)
    

        # Crea il layout principale
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.input_textbox)
        main_layout.addLayout(sidebar_layout)
        main_layout.addWidget(self.output_textbox)

        # Crea il widget principale e imposta il layout
        main_widget = QWidget()
        main_widget.setLayout(main_layout)

        # Imposta il widget principale come widget centrale della finestra
        self.setCentralWidget(main_widget)

    def clear_text(self):
        self.input_textbox.clear()
        self.output_textbox.clear()

    def set_num_samples(self):
        self.generator.num_samples = self.num_samples_spinbox.value()

    def set_max_new_tokens(self):
        self.generator.max_new_tokens = self.max_new_tokens_spinbox.value()

    def set_temperature(self):
        self.generator.temperature = self.temperature_spinbox.value()

    def set_top_k(self):
        val = self.top_k_spinbox.value()
        if val == 0:
            self.generator.top_k = None
        else: 
            self.generator.top_k = val
    
    def change_model(self):
        self.generator.set_init_from(self.model_box.currentText())

    def generate_text(self):
        self.output_textbox.clear()
        
        input = pp.preprocess_input(self.input_textbox.toPlainText())
        output_message=""
        output_message+=f'----INPUT-----\n'
        output_message+=f'{input}\n'
        output_message+=f'--------------\n'
        
        
        samples = self.generator.generate(input)
        k = 1
        for sample in samples:
            sample = sample.replace(input, "", 1)
            output_message+=f'{k}-----------\n'
            output_message+=f'{decode_java_code(sample)}\n'
            output_message+=f'--------------\n'
            k+=1
        self.output_textbox.setText(output_message)
    


def decode_java_code(encoded_text):
    encoded_text = encoded_text.strip().lstrip("<s>").rstrip("</s>")
    # Parse the one-line code using the encoder
    tokens = encoded_text.split(" ")
    # Initialize variables for tracking the indentation level and output string
    indent_level = 0
    output = ""
    # Iterate through the tokens and recreate the Java code
    for i, token in enumerate(tokens):
        # Handle special cases where the token affects the indentation level
        if token in ["{", "}"]:
            if token in ["}"]:
                indent_level -= 1
            if token in ["{"]:
                indent_level += 1
            output += token + "\n" + "  " * indent_level
            continue
        elif token == ";":
            output += token + "\n" + "  " * indent_level
            continue
        # Replace any placeholders for literals with their actual values
        if token.startswith("<"):
            literal_type, *literal_val = token[1:-1].split(":")
            literal_val = literal_val[0] if literal_val else None
            if literal_val:
                if literal_type == "STR_LIT":
                    token = '"' + literal_val + '"'
                elif literal_type == "CHAR_LIT":
                    token = "'" + literal_val + "'"
                elif literal_type == "NUM_LIT":
                    token = literal_val
            else:
                if literal_type == "STR_LIT":
                    token = '<STR_LIT>'
                elif literal_type == "CHAR_LIT":
                    token = "<CHAR_LIT>"
                elif literal_type == "NUM_LIT":
                    token = "<NUM_LIT>"
        # Add the token to the output string with appropriate indentation
        if token.endswith("."):
            output += token
        else:
            output += token + " "
    return output
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
