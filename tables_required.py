import mysql.connector
conn=mysql.connector.connect(user="root",password="",host="localhost",port="3306",database="project_4")
cur=conn.cursor()
cur.execute("create table if not exists admins(adminid real primary key, email varchar(30),a_name varchar(30),loggedin BOOLEAN)")
cur.execute("create table if not exists users(userid real primary key,u_name varchar(30), email varchar(30),loggedin BOOLEAN)")
cur.execute("create table if not exists algos(algoid real primary key,al_name varchar(30))")
cur.execute("create table if not exists interface(i_id real primary key,i_name varchar(30),algoid real references algos )")
cur.execute("create table if not exists data( id real primary key,userid real NOT NULL,data varchar(30),isadmin BOOLEAN,algoid real refernces algos,activity date NOT NULL)")
