#!/usr/bin/env python3
"""
populate_aromas.py
Genera aromi per ogni vino nel CSV usando Claude Haiku.
"""

import os
import sys
import pandas as pd
import httpx
from typing import Optional

# Config
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = "claude-haiku-4-5-20251001"
TIMEOUT = 5.0
CSV_PATH = "../data/wines.normalized.csv"

AROMAS_SYSTEM_PROMPT = """Sei un sommelier AIS esperto in analisi sensoriale.
Dato un vino (nome, denominazione, vitigni, regione), genera una lista di 3-5 aromi caratteristici.

Usa SOLO questi aromi canonici (separati da virgola):
- Frutta: agrumi, frutta_rossa, frutta_nera, ciliegia, fragola, lampone, mora, mirtillo, pesca, albicocca
- Fiori: fiori, rosa, violetta, gelsomino, acacia
- Spezie: spezie, anice, pepe, cannella, vaniglia
- Terziario: tostato, cuoio, tabacco, balsamico, minerale

Formato output: "ciliegia, spezie, cuoio" (solo nomi separati da virgola, nessun altro testo)

Esempi:
Barolo DOCG, nebbiolo, Piemonte → "rosa, cuoio, tostato, spezie"
Prosecco DOC, glera, Veneto → "fiori, frutta_bianca, agrumi"
Primitivo di Manduria, primitivo, Puglia → "frutta_nera, spezie, vaniglia"
"""


def generate_aromas_llm(name: str, denomination: str, grapes: str, region: str) -> Optional[str]:
    """Genera aromi per un vino usando Claude Haiku."""
    if not API_KEY:
        print("❌ ANTHROPIC_API_KEY non trovata")
        return None
    
    # Prepara contesto
    context_parts = []
    if name:
        context_parts.append(f"Nome: {name}")
    if denomination:
        context_parts.append(f"Denominazione: {denomination}")
    if grapes:
        context_parts.append(f"Vitigni: {grapes}")
    if region:
        context_parts.append(f"Regione: {region}")
    
    if not context_parts:
        return None
    
    context = "\n".join(context_parts)
    user_message = f"""Vino da analizzare:
{context}

Genera aromi caratteristici (3-5 aromi canonici separati da virgola):"""
    
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": MODEL,
                    "max_tokens": 100,
                    "system": AROMAS_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_message}],
                },
            )
        
        response.raise_for_status()
        data = response.json()
        aromas = data["content"][0]["text"].strip()
        
        # Pulizia: rimuovi virgolette, spazi extra
        aromas = aromas.replace('"', '').replace("'", "").strip()
        
        # Validazione: max 10 aromi, min 2
        aromas_list = [a.strip() for a in aromas.split(",") if a.strip()]
        if 2 <= len(aromas_list) <= 10:
            return ", ".join(aromas_list)
        else:
            print(f"⚠️  Aromi invalidi ({len(aromas_list)} aromi): {aromas}")
            return None
    
    except Exception as e:
        print(f"❌ Errore LLM: {e}")
        return None


def main():
    print("🍷 Populate Aromas - LLM Batch Generator")
    print("=" * 50)
    
    # Leggi CSV
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV non trovato: {CSV_PATH}")
        sys.exit(1)
    
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    print(f"✅ CSV caricato: {len(df)} vini")
    
    # Assicurati che esista colonna aromas
    if "aromas" not in df.columns:
        df["aromas"] = ""
        print("➕ Colonna 'aromas' creata")
    
    # Conta vini senza aromi
    empty_aromas = df[df["aromas"].str.strip() == ""]
    total_empty = len(empty_aromas)
    
    if total_empty == 0:
        print("✅ Tutti i vini hanno già aromi!")
        return
    
    print(f"📝 {total_empty} vini senza aromi da popolare")
    print()
    
    # Chiedi conferma
    confirm = input(f"Vuoi generare aromi per {total_empty} vini? (y/n): ").strip().lower()
    if confirm != "y":
        print("❌ Operazione annullata")
        return
    
    print()
    print("🚀 Inizio generazione...")
    print()
    
    # Genera aromi per ogni vino vuoto
    success_count = 0
    skip_count = 0
    
    for idx, row in empty_aromas.iterrows():
        wine_id = row.get("id", "")
        name = row.get("name", "")
        denom = row.get("denomination", "")
        grapes = row.get("grape_varieties", "")
        region = row.get("region", "")
        
        print(f"[{idx+1}/{len(df)}] {name[:40]:<40}", end=" → ")
        
        # Genera aromi
        aromas = generate_aromas_llm(name, denom, grapes, region)
        
        if aromas:
            df.at[idx, "aromas"] = aromas
            success_count += 1
            print(f"✅ {aromas}")
        else:
            skip_count += 1
            print("⚠️  Skip")
    
    print()
    print("=" * 50)
    print(f"✅ Successo: {success_count}")
    print(f"⚠️  Skip: {skip_count}")
    print()
    
    # Salva CSV
    backup_path = CSV_PATH.replace(".csv", ".backup.csv")
    df.to_csv(backup_path, index=False)
    print(f"💾 Backup salvato: {backup_path}")
    
    df.to_csv(CSV_PATH, index=False)
    print(f"💾 CSV aggiornato: {CSV_PATH}")
    print()
    print("🎉 Completato!")


if __name__ == "__main__":
    main()
