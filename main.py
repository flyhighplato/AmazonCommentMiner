
import java
import javax.swing as swing
import java.awt as awt


def appMain():
    win = swing.JFrame("Jython", size=(200, 200),windowClosing=exit)
    win.contentPane.layout = awt.FlowLayout(  )
    
    field = swing.JTextField(preferredSize=(200,20))
    field.setText("Hello world!")
    field.setEnabled(False);
    
    win.contentPane.add(field)
    
    win.pack()
    win.show()

if __name__ ==  "__main__":     
    appMain()
