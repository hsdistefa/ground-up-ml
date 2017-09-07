import os
import IPython.lib

c.NotebookApp.ip = '*'
c.NotebookApp.port = int(os.getenv('PORT', 8888))
c.NotebookApp.open_browser = False
c.NotebookApp.allow_root = True

# sets a password if PASSWORD environment variable is set
if 'PASSWORD' in os.environ:
    password = os.environ['PASSWORD']
    if password:
        c.NotebookApp.password = IPython.lib.passwd(password)
    else:
        c.NotebookApp.password = ''
        c.NotebookApp.token = ''
    del os.environ['PASSWORD']
