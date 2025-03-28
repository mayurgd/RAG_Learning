#That this line means that python will be our base. Can some comment here please and explain this line more?
FROM python:3.12 as base

#create a working directory in the virtual machine (VM)
WORKDIR /app

# copy all the python requirements stored in requirements.txt into the new directoy (in the VM)
COPY ./requirements.txt /app/requirements.txt

# activate the package manager pip. But why use no-cache-dir?
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# copy all files to the new directory (in the VM)
COPY ./ /app/
