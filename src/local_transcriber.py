# src/local_transcriber.py

import whisper
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transcribir_con_whisper(audio_path, modelo="base", idioma="es"):
    """
    Transcribe un archivo de audio usando OpenAI Whisper
    
    Args:
        audio_path (str): Ruta al archivo de audio
        modelo (str): Modelo de Whisper a usar (tiny, base, small, medium, large)
        idioma (str): Idioma del audio (es, en, etc.)
    
    Returns:
        str: Texto transcrito o None si hay error
    """
    try:
        # Verificar que el archivo existe
        if not os.path.exists(audio_path):
            logger.error(f"El archivo de audio no existe: {audio_path}")
            return None
        
        # Cargar modelo de Whisper
        logger.info(f"Cargando modelo Whisper: {modelo}")
        model = whisper.load_model(modelo)
        
        # Transcribir audio
        logger.info(f"Transcribiendo audio: {audio_path}")
        result = model.transcribe(audio_path, language=idioma)
        
        # Extraer texto
        texto_transcrito = result["text"].strip()
        
        if texto_transcrito:
            logger.info(f"Transcripción completada. Longitud: {len(texto_transcrito)} caracteres")
            return texto_transcrito
        else:
            logger.warning("La transcripción está vacía")
            return None
            
    except Exception as e:
        logger.error(f"Error al transcribir con Whisper: {str(e)}")
        return None

def transcribir_con_segmentos(audio_path, modelo="base", idioma="es"):
    """
    Transcribe un archivo de audio y devuelve segmentos con timestamps
    
    Args:
        audio_path (str): Ruta al archivo de audio
        modelo (str): Modelo de Whisper a usar
        idioma (str): Idioma del audio
    
    Returns:
        list: Lista de segmentos con timestamps
    """
    try:
        # Verificar que el archivo existe
        if not os.path.exists(audio_path):
            logger.error(f"El archivo de audio no existe: {audio_path}")
            return []
        
        # Cargar modelo de Whisper
        logger.info(f"Cargando modelo Whisper: {modelo}")
        model = whisper.load_model(modelo)
        
        # Transcribir audio
        logger.info(f"Transcribiendo audio con segmentos: {audio_path}")
        result = model.transcribe(audio_path, language=idioma)
        
        # Extraer segmentos
        segmentos = []
        for segment in result["segments"]:
            segmentos.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            })
        
        logger.info(f"Transcripción completada. Segmentos: {len(segmentos)}")
        return segmentos
        
    except Exception as e:
        logger.error(f"Error al transcribir con segmentos: {str(e)}")
        return []

def verificar_whisper_instalado():
    """
    Verifica que Whisper esté instalado correctamente
    
    Returns:
        bool: True si está instalado, False si no
    """
    try:
        import whisper
        # Intentar cargar el modelo más pequeño
        model = whisper.load_model("tiny")
        return True
    except ImportError:
        logger.error("OpenAI Whisper no está instalado")
        return False
    except Exception as e:
        logger.error(f"Error al verificar Whisper: {str(e)}")
        return False

def obtener_modelos_disponibles():
    """
    Obtiene la lista de modelos Whisper disponibles
    
    Returns:
        list: Lista de modelos disponibles
    """
    return ["tiny", "base", "small", "medium", "large"]

def obtener_idiomas_soportados():
    """
    Obtiene la lista de idiomas soportados por Whisper
    
    Returns:
        dict: Diccionario con códigos de idioma y nombres
    """
    return {
        "es": "Español",
        "en": "Inglés",
        "fr": "Francés",
        "de": "Alemán",
        "it": "Italiano",
        "pt": "Portugués",
        "ru": "Ruso",
        "ja": "Japonés",
        "ko": "Coreano",
        "zh": "Chino"
    }