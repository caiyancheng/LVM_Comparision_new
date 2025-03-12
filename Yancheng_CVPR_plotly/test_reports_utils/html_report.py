import os.path
from PIL import Image
import numpy as np

import matplotlib

class html_report:

    def __init__(self, file_name, header_file=None):

        self.name_ind = 0  # to generate unique file names        
        self.path, file = os.path.split(file_name)

        self.thead_open = False
        self.tbody_open = False

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        self.html_file_name = file_name
        self.fh = open(file_name, "w")

        self.println( "<html>")
        self.println( "<head>")
        if header_file:
            self.insert_file(header_file)
        self.println( "</head>")
        self.println( "<body>")

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def print(self, str):
        self.fh.write( str )

    def println(self, str):
        self.fh.write( str + "\n" )

    def insert_file(self, fname):
        with open(fname) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.println(line)        

    def insert_image(self, image, mousedown_image=None, class_tag=None):

        id_name = os.path.splitext(os.path.normpath(image).split(os.path.sep)[-1])[0]
        extra_tags = ''
        if not mousedown_image is None:
            extra_tags += 'data-original="{image}" data-alternate="{md_image}"'.format(md_image=mousedown_image, image=image)
        if not class_tag is None:
            extra_tags += ' class="{class_tag}"'.format(class_tag=class_tag)

        self.println( '<img src="{image}" alt="{id_name}" {ext_tags}/>'.format( ext_tags=extra_tags, image=image, id_name=id_name ) )

    def insert_video(self, video, mousedown_video=None, class_tag=None):

        id_name = os.path.splitext(os.path.normpath(video).split(os.path.sep)[-1])[0]
        extra_tags = ''
        if not mousedown_video is None:
            extra_tags += 'data-original="{video}" data-alternate="{md_video}"'.format(md_video=mousedown_video, video=video)
        if not class_tag is None:
            extra_tags += ' class="{class_tag}"'.format(class_tag=class_tag)

        self.println( '<video alt="{id_name}" {ext_tags} autoplay muted loop playsinline preload="auto"> <source src="{video}" type="video/mp4"> Your browser does not support the video tag </video>'.format( ext_tags=extra_tags, video=video, id_name=id_name ) )

    def insert_text(self, text):
        self.println("<p> {text} </p>".format(text=text))

    def insert_separator(self):
        self.println('<hr class="test_separator">')

    def insert_header(self, text):
        self.println('<h2>{text}</h2>'.format(text=text))

    def beg_div(self, id_tag=None, class_tag=None):
        self.print("<div")
        if id_tag:
            self.print(" id={id_tag}".format(id_tag=id_tag))
        if class_tag:
            self.print(" class={class_tag}".format(class_tag=class_tag))
        self.println(">")

    def close_div(self):
        self.println("</div>")

    def beg_p(self, id_tag=None, class_tag=None):
        self.print("<p")
        if id_tag:
            self.print(" id={id_tag}".format(id_tag=id_tag))
        if class_tag:
            self.print(" class={class_tag}".format(class_tag=class_tag))
        self.println(">")

    def close_p(self):
        self.println("</p>")

    def beg_table(self, id_tag=None, class_tag=None):
        self.print("<table")
        if id:
            self.print(" id={id}".format(id=id_tag))
        if class_tag:
            self.print(" class={}".format(class_tag))
        self.print(">")

    def end_table(self):
        if self.tbody_open:
            self.println("  </tbody>")
            self.tbody_open=False
        self.println("</table>")

    def beg_table_row(self, ishead=False):
        if ishead:
            if not self.thead_open:
                self.println("  <thead>")      
                self.thead_open=True
        else:
            if self.thead_open:
                self.println("  </thead>")      
                self.thead_open=False

            if not self.tbody_open:
                self.println("  <tbody>")      
                self.tbody_open=True

        self.println("    <tr>")      

    def end_table_row(self):
        self.println("    </tr>")      

    def beg_table_cell(self):
        if self.thead_open:
            self.println("      <th>")      
        else:
            self.println("      <td>")      

    def end_table_cell(self):
        if self.thead_open:
            self.println("      </th>")      
        else:
            self.println("      </td>")      

    def insert_table_cells(self,*args):
        if self.thead_open:
            tag="th"
        else:
            tag="td"
        for cell in args:
            self.println( "<{tag}>{cell}</{tag}>".format(tag=tag, cell=cell))

    def close( self, js_file=None ):

        if js_file is not None:
            self.println('<script type="text/javascript" src="js_files/'+js_file+'"></script>')

        self.insert_file(os.path.join(os.path.dirname(__file__), "html_preable.html"))
        self.println( "</body>")
        self.println( "</html>")
        self.fh.close()
        print( "HTML report generated in file://{}".format(os.path.abspath(self.html_file_name)) )

