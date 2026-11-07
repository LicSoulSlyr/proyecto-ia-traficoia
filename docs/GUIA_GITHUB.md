# Guía para crear el repositorio (entregable del sábado)

## Quien coordina (Integrante 1 o 4)

1. En GitHub: **New repository** → nombre `proyecto-ia-traficoia` → privado → *sin* README (ya lo tenemos).
2. **Settings → Collaborators → Add people**: los otros 3 integrantes y el docente.
3. En la terminal, dentro de la carpeta `proyecto_ia/`:

```bash
git init
git branch -M main
git add .
git commit -m "feat: estructura inicial, entorno y scripts de datos"
git remote add origin https://github.com/USUARIO/proyecto-ia-traficoia.git
git push -u origin main
```

## Cada integrante

```bash
git clone https://github.com/USUARIO/proyecto-ia-traficoia.git
cd proyecto-ia-traficoia
conda env create -f environment.yml
conda activate proyecto_ia
python src/verificar_entorno.py     # tomar captura de pantalla
git config user.name  "Tu Nombre"
git config user.email "tu_correo@ejemplo.com"
```

## Para que haya commits de los 4 integrantes esta semana

| Integrante | Commit sugerido |
|---|---|
| 1 | Ajustar `README.md` con nombres del equipo y rol de cada quien |
| 2 | Ejecutar la descarga, completar fecha de revisión de licencias en `data/README.md` y subir `data/raw/manifest.json` |
| 3 | Subir `reports/captura_entorno_integrante3.png` y probar `conda env create` desde cero |
| 4 | Subir `reports/captura_entorno_integrante4.png` y revisar `.gitignore` |

Flujo diario: `git pull` → trabajar → `git add` → `git commit -m "mensaje claro"` → `git push`.
No subas datos pesados ni modelos grandes (el `.gitignore` ya los excluye).

## Exportar el entorno final (cuando todos lo tengan funcionando)

```bash
conda env export --from-history > environment.yml
pip freeze > requirements.txt
```

Revisa `environment.yml` después: `--from-history` no incluye los paquetes de la sección `pip`
(torch, gymnasium, etc.). Si se pierden, vuelve a agregarlos a mano.
