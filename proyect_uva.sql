PGDMP  
                	    |            proyect_uva    16.4    16.4                0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                      false                       0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                      false                       0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                      false                       1262    16403    proyect_uva    DATABASE     �   CREATE DATABASE proyect_uva WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'English_United States.1252';
    DROP DATABASE proyect_uva;
                usuario_uva    false                       0    0    SCHEMA public    ACL     +   GRANT ALL ON SCHEMA public TO usuario_uva;
                   pg_database_owner    false    5            �            1259    16449 	   consultas    TABLE     �   CREATE TABLE public.consultas (
    id integer NOT NULL,
    user_id integer,
    class_name character varying(100),
    consulta_fecha timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);
    DROP TABLE public.consultas;
       public         heap    usuario_uva    false            �            1259    16448    consultas_id_seq    SEQUENCE     �   CREATE SEQUENCE public.consultas_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 '   DROP SEQUENCE public.consultas_id_seq;
       public          usuario_uva    false    220                       0    0    consultas_id_seq    SEQUENCE OWNED BY     E   ALTER SEQUENCE public.consultas_id_seq OWNED BY public.consultas.id;
          public          usuario_uva    false    219            �            1259    16408    usuario    TABLE     �   CREATE TABLE public.usuario (
    id integer NOT NULL,
    nombre character varying(100) NOT NULL,
    email character varying(100) NOT NULL,
    "contraseña" character varying(255) NOT NULL
);
    DROP TABLE public.usuario;
       public         heap    usuario_uva    false            �            1259    16411    usuario_id_seq    SEQUENCE     �   CREATE SEQUENCE public.usuario_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 %   DROP SEQUENCE public.usuario_id_seq;
       public          usuario_uva    false    215                       0    0    usuario_id_seq    SEQUENCE OWNED BY     A   ALTER SEQUENCE public.usuario_id_seq OWNED BY public.usuario.id;
          public          usuario_uva    false    216            �            1259    16412    usuarios    TABLE     0  CREATE TABLE public.usuarios (
    id integer NOT NULL,
    nombre character varying(255) NOT NULL,
    correo character varying(255) NOT NULL,
    "contraseña" character varying(255) NOT NULL,
    fecha_registro timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    estado boolean DEFAULT true
);
    DROP TABLE public.usuarios;
       public         heap    postgres    false            	           0    0    TABLE usuarios    ACL     K   GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.usuarios TO usuario_uva;
          public          postgres    false    217            �            1259    16419    usuarios_id_seq    SEQUENCE     �   CREATE SEQUENCE public.usuarios_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 &   DROP SEQUENCE public.usuarios_id_seq;
       public          postgres    false    217            
           0    0    usuarios_id_seq    SEQUENCE OWNED BY     C   ALTER SEQUENCE public.usuarios_id_seq OWNED BY public.usuarios.id;
          public          postgres    false    218                       0    0    SEQUENCE usuarios_id_seq    ACL     F   GRANT SELECT,USAGE ON SEQUENCE public.usuarios_id_seq TO usuario_uva;
          public          postgres    false    218            ^           2604    16452    consultas id    DEFAULT     l   ALTER TABLE ONLY public.consultas ALTER COLUMN id SET DEFAULT nextval('public.consultas_id_seq'::regclass);
 ;   ALTER TABLE public.consultas ALTER COLUMN id DROP DEFAULT;
       public          usuario_uva    false    220    219    220            Z           2604    16421 
   usuario id    DEFAULT     h   ALTER TABLE ONLY public.usuario ALTER COLUMN id SET DEFAULT nextval('public.usuario_id_seq'::regclass);
 9   ALTER TABLE public.usuario ALTER COLUMN id DROP DEFAULT;
       public          usuario_uva    false    216    215            [           2604    16422    usuarios id    DEFAULT     j   ALTER TABLE ONLY public.usuarios ALTER COLUMN id SET DEFAULT nextval('public.usuarios_id_seq'::regclass);
 :   ALTER TABLE public.usuarios ALTER COLUMN id DROP DEFAULT;
       public          postgres    false    218    217            �          0    16449 	   consultas 
   TABLE DATA           L   COPY public.consultas (id, user_id, class_name, consulta_fecha) FROM stdin;
    public          usuario_uva    false    220   !       �          0    16408    usuario 
   TABLE DATA           C   COPY public.usuario (id, nombre, email, "contraseña") FROM stdin;
    public          usuario_uva    false    215   �"       �          0    16412    usuarios 
   TABLE DATA           ]   COPY public.usuarios (id, nombre, correo, "contraseña", fecha_registro, estado) FROM stdin;
    public          postgres    false    217   #                  0    0    consultas_id_seq    SEQUENCE SET     ?   SELECT pg_catalog.setval('public.consultas_id_seq', 33, true);
          public          usuario_uva    false    219                       0    0    usuario_id_seq    SEQUENCE SET     =   SELECT pg_catalog.setval('public.usuario_id_seq', 1, false);
          public          usuario_uva    false    216                       0    0    usuarios_id_seq    SEQUENCE SET     >   SELECT pg_catalog.setval('public.usuarios_id_seq', 13, true);
          public          postgres    false    218            i           2606    16455    consultas consultas_pkey 
   CONSTRAINT     V   ALTER TABLE ONLY public.consultas
    ADD CONSTRAINT consultas_pkey PRIMARY KEY (id);
 B   ALTER TABLE ONLY public.consultas DROP CONSTRAINT consultas_pkey;
       public            usuario_uva    false    220            a           2606    16426    usuario usuario_email_key 
   CONSTRAINT     U   ALTER TABLE ONLY public.usuario
    ADD CONSTRAINT usuario_email_key UNIQUE (email);
 C   ALTER TABLE ONLY public.usuario DROP CONSTRAINT usuario_email_key;
       public            usuario_uva    false    215            c           2606    16428    usuario usuario_pkey 
   CONSTRAINT     R   ALTER TABLE ONLY public.usuario
    ADD CONSTRAINT usuario_pkey PRIMARY KEY (id);
 >   ALTER TABLE ONLY public.usuario DROP CONSTRAINT usuario_pkey;
       public            usuario_uva    false    215            e           2606    16430    usuarios usuarios_correo_key 
   CONSTRAINT     Y   ALTER TABLE ONLY public.usuarios
    ADD CONSTRAINT usuarios_correo_key UNIQUE (correo);
 F   ALTER TABLE ONLY public.usuarios DROP CONSTRAINT usuarios_correo_key;
       public            postgres    false    217            g           2606    16432    usuarios usuarios_pkey 
   CONSTRAINT     T   ALTER TABLE ONLY public.usuarios
    ADD CONSTRAINT usuarios_pkey PRIMARY KEY (id);
 @   ALTER TABLE ONLY public.usuarios DROP CONSTRAINT usuarios_pkey;
       public            postgres    false    217            j           2606    16456     consultas consultas_user_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.consultas
    ADD CONSTRAINT consultas_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.usuarios(id);
 J   ALTER TABLE ONLY public.consultas DROP CONSTRAINT consultas_user_id_fkey;
       public          usuario_uva    false    217    220    4711            �   �  x���=nA�k�{�x���Li e��L�X�@��)�k�K닅#ō��U����<��!�|:w���F�;�;��N�Qk�j�W8�:��� ��r����q����v���c{Y�H��-�-Z�� ���;��M	��y��P�@�z��Z��@��������ez8<���4Z�VEں���]�h�h��=�v֢�5I4�>&�n��
��	������j	r�
$K���h�i{-Θ�B��B��-h�sV
��;�ߤݺH	w�󉾃��!V�Xǈ�P#:E������=rY��Zt�����4.^#L�O�}��ߎ�K�u�&Ss)�3�XrO-M��yNB!t�tq%����:��҇i�G����_XL)/P��}�
��mB�Lㇽ6�,i��g0�4J�V=@��r͍Sq�O���QFb�q2=Ϡu�Tl���.A�[��D�<� �?[�]      �      x������ � �      �   �  x���Kk�0���_��\#43�f�S ���z�P(�,�o�M~}eo(��! �3���@=����Z����@���pܥM��S�d��`n��!j��� ��ۻ�U���Y���ً�4�S�ߔ^�f�Wy���i��s�yn.�I�� "��"����5�e�&�v�/~��eF���2QB�*�����$�etN��~�+��/j����.���X����P^�j;m2��uG쇊c�H���O�]yQy�.��� &NK3�z� �Zb��E�4��h0�+|��ʁH����`rA�\���,�����{�OH̧�O�^n	u^(��Ƹ��H���v�թ[���Re���I��X���狼K?8w���ö�Xj�o��ȳ���qq���.�Љ�-����@�J#���ˉڦ��B�k��ڮN@n)K�L�S灂MŶ��E�������(c��WY-Ďo�^,� ��     