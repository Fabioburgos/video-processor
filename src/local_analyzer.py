import requests
import json
import os
import logging
import time
import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"

# Configura tu clave API de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("⚠️ OPENAI_API_KEY no está configurada. Algunas funciones pueden fallar.")

def diagnosticar_ollama():
    """
    Función de diagnóstico completo para Ollama
    
    Returns:
        dict: Resultado del diagnóstico
    """
    diagnostico = {
        "conexion": False,
        "modelos": [],
        "modelo_recomendado": None,
        "errores": []
    }
    
    # 1. Verificar conexión básica
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            diagnostico["conexion"] = True
            logger.info("✅ Conexión con Ollama exitosa")
        else:
            diagnostico["errores"].append(f"Error de conexión: {response.status_code}")
            logger.error(f"❌ Error de conexión: {response.status_code}")
            return diagnostico
    except requests.exceptions.RequestException as e:
        diagnostico["errores"].append(f"No se puede conectar con Ollama: {str(e)}")
        logger.error(f"❌ No se puede conectar con Ollama: {str(e)}")
        return diagnostico
    
    # 2. Obtener modelos disponibles
    try:
        data = response.json()
        modelos = [model["name"] for model in data.get("models", [])]
        diagnostico["modelos"] = modelos
        logger.info(f"✅ Modelos disponibles: {modelos}")
        
        # Buscar modelo recomendado
        modelos_recomendados = ["llama3", "llama2", "mistral", "phi", "codellama"]
        for modelo in modelos_recomendados:
            if any(modelo in m for m in modelos):
                diagnostico["modelo_recomendado"] = next(m for m in modelos if modelo in m)
                break
        
        if not diagnostico["modelo_recomendado"] and modelos:
            diagnostico["modelo_recomendado"] = modelos[0]
            
    except Exception as e:
        diagnostico["errores"].append(f"Error al obtener modelos: {str(e)}")
        logger.error(f"❌ Error al obtener modelos: {str(e)}")
    
    return diagnostico

def instalar_modelo_si_necesario(modelo="llama3"):
    """
    Instala un modelo si no está disponible
    
    Args:
        modelo (str): Nombre del modelo a instalar
    
    Returns:
        bool: True si el modelo está disponible, False si no
    """
    try:
        # Verificar si ya está instalado
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            modelos = [model["name"] for model in data.get("models", [])]
            
            # Buscar modelo exacto o similar
            if any(modelo in m for m in modelos):
                logger.info(f"✅ Modelo {modelo} ya está instalado")
                return True
        
        # Instalar modelo
        logger.info(f"📥 Instalando modelo {modelo}...")
        
        data = {"name": modelo}
        response = requests.post(
            f"{OLLAMA_URL}/api/pull",
            json=data,
            timeout=600  # 10 minutos timeout
        )
        
        if response.status_code == 200:
            logger.info(f"✅ Modelo {modelo} instalado exitosamente")
            return True
        else:
            logger.error(f"❌ Error al instalar modelo {modelo}: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error al instalar modelo {modelo}: {str(e)}")
        return False

def analizar_con_ollama_mejorado(modelo, ruta_prompt, transcripcion_texto):
    """
    Versión mejorada del análisis usando OpenAI
    
    Args:
        modelo (str): Nombre del modelo de OpenAI
        ruta_prompt (str): Ruta al archivo de prompt
        transcripcion_texto (str): Texto a analizar
    
    Returns:
        str: Análisis generado o None si hay error
    """
    try:
        diagnostico = diagnosticar_ollama()
        if not diagnostico["conexion"]:
            logger.error("❌ No hay conexión con OpenAI")
            return None

        modelo_usar = modelo
        if modelo not in diagnostico["modelos"]:
            logger.warning(f"⚠️ Modelo {modelo} no está en la lista recomendada. Usando recomendado.")
            modelo_usar = diagnostico["modelo_recomendado"]

        # Configurar cliente OpenAI
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        prompt_sistema = leer_prompt_mejorado(ruta_prompt)
        if not prompt_sistema:
            logger.error("❌ No se pudo leer el prompt")
            return None

        mensaje_completo = f"""{prompt_sistema}

TRANSCRIPCIÓN A ANALIZAR:
{transcripcion_texto[:3000]}...

Por favor, analiza la transcripción y proporciona calificaciones numéricas claras."""

        messages = [
            {"role": "system", "content": "Eres un asistente que evalúa sesiones educativas con precisión."},
            {"role": "user", "content": mensaje_completo}
        ]

        max_intentos = 3
        for intento in range(max_intentos):
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1500,
                    top_p=0.9
                )
                texto_analisis = response.choices[0].message.content.strip()
                if texto_analisis:
                    logger.info(f"✅ Análisis completado (intento {intento + 1})")
                    logger.info(f"📊 Longitud del análisis: {len(texto_analisis)} caracteres")
                    return texto_analisis
                else:
                    logger.warning(f"⚠️ Respuesta vacía en intento {intento + 1}")

            except openai.OpenAIError as e:
                logger.error(f"❌ Error de OpenAI en intento {intento + 1}: {str(e)}")

            time.sleep(2)

        logger.error("❌ Se agotaron todos los intentos")
        return None

    except Exception as e:
        logger.error(f"❌ Error inesperado en análisis: {str(e)}")
        return None

def leer_prompt_mejorado(ruta_prompt):
    """
    Versión mejorada para leer prompts con fallback
    
    Args:
        ruta_prompt (str): Ruta al archivo de prompt
    
    Returns:
        str: Contenido del prompt
    """
    try:
        if ruta_prompt and os.path.exists(ruta_prompt):
            with open(ruta_prompt, "r", encoding="utf-8") as f:
                contenido = f.read().strip()
                if contenido:
                    logger.info(f"✅ Prompt leído desde: {ruta_prompt}")
                    return contenido
        
        logger.info("🔧 Usando prompt por defecto")
        return obtener_prompt_optimizado()
        
    except Exception as e:
        logger.error(f"❌ Error al leer prompt: {str(e)}")
        return obtener_prompt_optimizado()

def obtener_prompt_optimizado():
    """
    Prompt optimizado para obtener respuestas estructuradas
    
    Returns:
        str: Prompt optimizado
    """
    return """Analiza la siguiente transcripción de tutoría y proporciona calificaciones numéricas del 1 al 10:

ASPECTOS A EVALUAR:
1. Claridad en la explicación
2. Participación del estudiante  
3. Uso de ejemplos y recursos
4. Resolución de dudas
5. Ambiente de aprendizaje

FORMATO REQUERIDO:
Claridad en la explicación: 8/10
Participación del estudiante: 7/10
Uso de ejemplos y recursos: 6/10
Resolución de dudas: 9/10
Ambiente de aprendizaje: 8/10

JUSTIFICACIÓN:
[Explicación breve de cada calificación]

RECOMENDACIONES:
[3-5 recomendaciones específicas para mejorar]

PROMEDIO GENERAL: 7.6/10"""

def probar_ollama_basico():
    """
    Prueba básica de funcionamiento de Ollama
    
    Returns:
        dict: Resultado de la prueba
    """
    resultado = {
        "exito": False,
        "modelo_usado": None,
        "respuesta": None,
        "error": None
    }
    
    try:
        # Diagnóstico
        diagnostico = diagnosticar_ollama()
        if not diagnostico["conexion"]:
            resultado["error"] = "No hay conexión con Ollama"
            return resultado
        
        if not diagnostico["modelos"]:
            resultado["error"] = "No hay modelos instalados"
            return resultado
        
        modelo = diagnostico["modelo_recomendado"]
        resultado["modelo_usado"] = modelo
        
        # Prueba simple
        data = {
            "model": modelo,
            "prompt": "Responde solo con 'OK' si me entiendes.",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 10
            }
        }
        
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            respuesta_data = response.json()
            respuesta_texto = respuesta_data.get("response", "").strip()
            
            resultado["exito"] = True
            resultado["respuesta"] = respuesta_texto
            
        else:
            resultado["error"] = f"Error HTTP: {response.status_code}"
            
    except Exception as e:
        resultado["error"] = str(e)
    
    return resultado

def generar_comando_instalacion():
    """
    Genera comandos para instalar modelos recomendados
    
    Returns:
        list: Lista de comandos
    """
    comandos = [
        "ollama pull llama3",
        "ollama pull mistral", 
        "ollama pull phi",
        "ollama serve"
    ]
    
    return comandos

# Función para uso directo en el script principal
def analizar_con_ollama(modelo, ruta_prompt, transcripcion_texto):
    """
    Función principal para compatibilidad con el código existente
    """
    return analizar_con_ollama_mejorado(modelo, ruta_prompt, transcripcion_texto)