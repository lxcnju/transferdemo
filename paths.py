import os


try:
    from naie.context import Context
    print(Context.get_project_path())
    project_path = Context.get_project_path()
    data_dir = os.path.join(project_path, "Dataset", "transferdemo", "digits", "data")
    
except Exception:
    data_dir = "./pkls/"
