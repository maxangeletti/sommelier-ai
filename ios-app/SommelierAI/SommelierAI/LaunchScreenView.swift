//
//  LaunchScreenView.swift
//  SommelierAI
//
//  Created on 3 April 2026
//

import SwiftUI

struct LaunchScreenView: View {
    var body: some View {
        ZStack {
            // Background color (crema dalla splash)
            Color(red: 245/255, green: 235/255, blue: 217/255)
                .ignoresSafeArea()
            
            // Splash image - aspect fill per riempire tutto lo schermo
            Image("splash_def")
                .resizable()
                .aspectRatio(contentMode: .fill)
                .ignoresSafeArea()
        }
    }
}

#Preview {
    LaunchScreenView()
}
