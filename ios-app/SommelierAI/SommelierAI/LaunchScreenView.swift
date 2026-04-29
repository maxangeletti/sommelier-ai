//
//  LaunchScreenView.swift
//  SommelierAI
//
//  Created on 3 April 2026
//

import SwiftUI

struct LaunchScreenView: View {
    var body: some View {
        // ✅ Splash FULLSCREEN - copre tutto senza background
        Image("splash_def")
            .resizable()
            .scaledToFill()
            .ignoresSafeArea()
    }
}

#Preview {
    LaunchScreenView()
}
