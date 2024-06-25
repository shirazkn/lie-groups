import pickle
import os, shutil, sys


def add_item(path, item):
    with open(path, 'ab') as file:
        pickle.dump(item, file, pickle.HIGHEST_PROTOCOL)


def get_generator(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass

def read_all(path):
    items = []
    for item in get_generator(path):
        items.extend(item)
    
    return items


def file_exists(path):
    return os.path.exists(path)


def ask_delete(path, description = None):
    if file_exists(path) is False:
        return
    
    if description is None:
        description = ""

    else:
        description += "\n"

    description += f"Delete {path}? (y/n): "
    response = input(description) 
    if response.lower()[0] == "y": 
        os.remove(path)
        print("Deleted.") 

    elif response.lower()[0] == "n": 
        print("Exiting.") 
        sys.exit()

    else: 
        print("Please enter 'yes' or 'no'.\n")
        ask_delete(path, description)


def delete(path, warn = True):
    if file_exists(path):
        if warn:
            print(f"Deleting {path}...")

        try:
            os.remove(path)
        except:
            shutil.rmtree(path)
