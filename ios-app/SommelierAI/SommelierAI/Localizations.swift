//
//  Localizations.swift
//  SommelierAI
//
//  PURPOSE
//  Traduzione valori tecnici del dataset in italiano per la UI.
//  Il dataset e il motore di ranking restano invariati.
//

import Foundation

struct WineLocalizer {

    static func country(_ raw: String?) -> String {
        guard let raw = raw else { return "" }
        switch raw.lowercased() {
        case "italy":    return "Italia"
        case "france":   return "Francia"
        case "germany":  return "Germania"
        case "slovenia": return "Slovenia"
        case "spain":    return "Spagna"
        case "portugal": return "Portogallo"
        default:         return raw
        }
    }

    static func color(_ raw: String?) -> String {
        guard let raw = raw else { return "" }
        switch raw.lowercased() {
        case "red":          return "Rosso"
        case "white":        return "Bianco"
        case "rose":         return "Rosato"
        case "orange":       return "Orange"
        case "ruby_red":     return "Rosso rubino"
        case "ruby red":     return "Rosso rubino"
        case "straw_yellow": return "Giallo paglierino"
        case "straw yellow": return "Giallo paglierino"
        case "sparkling":    return "Spumante"
        case "sweet":        return "Dolce"
        default:             return raw.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    static func region(_ raw: String?) -> String {
        guard let raw = raw else { return "" }
        switch raw.lowercased() {
        case "bourgogne":  return "Borgogna"
        case "bordeaux":   return "Bordeaux"
        case "champagne":  return "Champagne"
        case "alsace":     return "Alsazia"
        case "loire":      return "Loira"
        case "rhone":      return "Rodano"
        case "tuscany":    return "Toscana"
        case "piedmont":   return "Piemonte"
        default:           return raw
        }
    }
    
    // ✅ NEW: Localizzazione intensità/tannicità/acidità
    static func intensity(_ raw: String?) -> String {
        guard let raw = raw else { return "" }
        switch raw.lowercased() {
        case "low":    return "Bassa"
        case "medium": return "Media"
        case "high":   return "Alta"
        default:       return raw.capitalized
        }
    }
    
    static func tannins(_ raw: String?) -> String {
        guard let raw = raw else { return "" }
        switch raw.lowercased() {
        case "low":          return "Bassa"
        case "medium":       return "Media"
        case "medium_plus":  return "Media-Alta"
        case "high":         return "Alta"
        default:             return raw.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }
    
    static func acidity(_ raw: String?) -> String {
        guard let raw = raw else { return "" }
        switch raw.lowercased() {
        case "low":          return "Bassa"
        case "medium":       return "Media"
        case "medium_plus":  return "Media-Alta"
        case "high":         return "Alta"
        default:             return raw.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }
}
