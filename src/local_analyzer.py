import openai
import requests
import json
import os
import logging
import time
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"

# Configura tu clave API de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("⚠️ OPENAI_API_KEY no está configurada. Solo funcionará modo Ollama.")

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

def analizar_con_ollama_puro(modelo, ruta_prompt, transcripcion_texto):
    """
    Versión que usa SOLO Ollama (sin OpenAI)
    
    Args:
        modelo (str): Nombre del modelo de Ollama
        ruta_prompt (str): Ruta al archivo de prompt
        transcripcion_texto (str): Texto a analizar
    
    Returns:
        str: Análisis generado o None si hay error
    """
    try:
        diagnostico = diagnosticar_ollama()
        if not diagnostico["conexion"]:
            logger.error("❌ No hay conexión con Ollama")
            return None

        modelo_usar = modelo
        if modelo not in diagnostico["modelos"]:
            if diagnostico["modelo_recomendado"]:
                logger.warning(f"⚠️ Modelo {modelo} no disponible. Usando {diagnostico['modelo_recomendado']}")
                modelo_usar = diagnostico["modelo_recomendado"]
            else:
                logger.error("❌ No hay modelos disponibles")
                return None

        prompt_sistema = leer_prompt_mejorado(ruta_prompt)
        if not prompt_sistema:
            logger.error("❌ No se pudo leer el prompt")
            return None

        mensaje_completo = f"""{prompt_sistema}

TRANSCRIPCIÓN A ANALIZAR:
{transcripcion_texto[:3000]}...

Por favor, analiza la transcripción y proporciona calificaciones numéricas claras."""

        max_intentos = 3
        for intento in range(max_intentos):
            try:
                data = {
                    "model": modelo_usar,
                    "prompt": mensaje_completo,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1500,
                        "top_p": 0.9
                    }
                }

                response = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json=data,
                    timeout=120
                )

                if response.status_code == 200:
                    respuesta_data = response.json()
                    texto_analisis = respuesta_data.get("response", "").strip()
                    
                    if texto_analisis:
                        logger.info(f"✅ Análisis completado (intento {intento + 1})")
                        logger.info(f"📊 Longitud del análisis: {len(texto_analisis)} caracteres")
                        return texto_analisis
                    else:
                        logger.warning(f"⚠️ Respuesta vacía en intento {intento + 1}")
                else:
                    logger.error(f"❌ Error HTTP {response.status_code} en intento {intento + 1}")

            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Error de conexión en intento {intento + 1}: {str(e)}")

            time.sleep(2)

        logger.error("❌ Se agotaron todos los intentos")
        return None

    except Exception as e:
        logger.error(f"❌ Error inesperado en análisis: {str(e)}")
        return None

def analizar_con_openai_puro(modelo, ruta_prompt, transcripcion_texto):
    """
    Versión que usa correctamente OpenAI
    
    Args:
        modelo (str): Nombre del modelo de OpenAI (ej: 'gpt-3.5-turbo', 'gpt-4')
        ruta_prompt (str): Ruta al archivo de prompt
        transcripcion_texto (str): Texto a analizar
    
    Returns:
        str: Análisis generado o None si hay error
    """
    try:
        if not OPENAI_API_KEY:
            logger.error("❌ OPENAI_API_KEY no está configurada")
            return None

        # Configurar cliente OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

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

        # Modelos válidos de OpenAI
        modelos_validos = ["gpt-4.1-nano","gpt-4o-mini"]
        modelo_usar = modelo if modelo in modelos_validos else "gpt-3.5-turbo"

        if modelo_usar != modelo:
            logger.warning(f"⚠️ Modelo {modelo} no válido. Usando {modelo_usar}")

        max_intentos = 3
        for intento in range(max_intentos):
            try:
                response = client.chat.completions.create(
                    model=modelo_usar,
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

def analizar_con_ollama_mejorado(modelo, ruta_prompt, transcripcion_texto):
    """
    Función inteligente que decide automáticamente entre Ollama y OpenAI
    
    Args:
        modelo (str): Nombre del modelo
        ruta_prompt (str): Ruta al archivo de prompt
        transcripcion_texto (str): Texto a analizar
    
    Returns:
        str: Análisis generado o None si hay error
    """
    # Primero intentar con Ollama si está disponible
    diagnostico = diagnosticar_ollama()
    if diagnostico["conexion"] and diagnostico["modelos"]:
        logger.info("🔧 Usando Ollama local...")
        return analizar_con_ollama_puro(modelo, ruta_prompt, transcripcion_texto)
    
    # Si Ollama no está disponible, intentar con OpenAI
    elif OPENAI_API_KEY:
        logger.info("🌐 Ollama no disponible, usando OpenAI...")
        return analizar_con_openai_puro(modelo, ruta_prompt, transcripcion_texto)
    
    # Si ninguno está disponible
    else:
        logger.error("❌ Ni Ollama ni OpenAI están disponibles")
        logger.error("💡 Opciones:")
        logger.error("   - Instalar y ejecutar Ollama: ollama serve")
        logger.error("   - Configurar OPENAI_API_KEY")
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

def probar_openai_basico():
    """
    Prueba básica de funcionamiento de OpenAI
    
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
        if not OPENAI_API_KEY:
            resultado["error"] = "OPENAI_API_KEY no está configurada"
            return resultado
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        modelo = "gpt-3.5-turbo"
        resultado["modelo_usado"] = modelo
        
        response = client.chat.completions.create(
            model=modelo,
            messages=[{"role": "user", "content": "Responde solo con 'OK' si me entiendes."}],
            max_tokens=10,
            temperature=0.1
        )
        
        respuesta_texto = response.choices[0].message.content.strip()
        resultado["exito"] = True
        resultado["respuesta"] = respuesta_texto
        
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
        "# Instalar Ollama",
        "curl -fsSL https://ollama.ai/install.sh | sh",
        "",
        "# Descargar modelos recomendados",
        "ollama pull llama3",
        "ollama pull mistral", 
        "ollama pull phi",
        "",
        "# Iniciar servidor",
        "ollama serve"
    ]
    
    return comandos

def mostrar_estado_sistema():
    """
    Muestra el estado actual del sistema
    """
    print("=" * 50)
    print("🔍 DIAGNÓSTICO DEL SISTEMA")
    print("=" * 50)
    
    # Verificar Ollama
    print("\n📱 OLLAMA:")
    diagnostico = diagnosticar_ollama()
    if diagnostico["conexion"]:
        print(f"✅ Conectado - Modelos: {len(diagnostico['modelos'])}")
        if diagnostico["modelos"]:
            print(f"   Modelos disponibles: {', '.join(diagnostico['modelos'])}")
            print(f"   Modelo recomendado: {diagnostico['modelo_recomendado']}")
    else:
        print("❌ No conectado")
        print("💡 Para instalar:")
        for cmd in generar_comando_instalacion():
            print(f"   {cmd}")
    
    # Verificar OpenAI
    print("\n🌐 OPENAI:")
    if OPENAI_API_KEY:
        print("✅ API Key configurada")
        # Hacer prueba rápida
        prueba = probar_openai_basico()
        if prueba["exito"]:
            print("✅ Conexión exitosa")
        else:
            print(f"❌ Error: {prueba['error']}")
    else:
        print("❌ API Key no configurada")
        print("💡 Para configurar:")
        print("   export OPENAI_API_KEY='tu-api-key-aqui'")
    
    print("=" * 50)

# Función para uso directo en el script principal (compatible con código existente)
def analizar_con_ollama(modelo, ruta_prompt, transcripcion_texto):
    """
    Función principal para compatibilidad con el código existente
    Ahora es inteligente y usa la mejor opción disponible
    """
    return analizar_con_ollama_mejorado(modelo, ruta_prompt, transcripcion_texto)

# Función de prueba para verificar que todo funciona
def main():
    """
    Función de prueba
    """
    mostrar_estado_sistema()
    
    # Prueba básica
    print("\n🧪 PRUEBA BÁSICA:")
    resultado = analizar_con_ollama_mejorado(
        "llama3", 
        None,  # Usar prompt por defecto
        "El profesor explicó matemáticas y el estudiante hizo preguntas."
    )
    
    if resultado:
        print("✅ Análisis exitoso:")
        print(resultado[:200] + "..." if len(resultado) > 200 else resultado)
    else:
        print("❌ No se pudo realizar el análisis")