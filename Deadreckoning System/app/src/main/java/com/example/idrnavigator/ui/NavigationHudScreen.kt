package com.example.idrnavigator.ui

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import kotlinx.coroutines.delay
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Fill
import androidx.compose.ui.unit.dp
import com.example.idrnavigator.ui.theme.*
import com.example.idrnavigator.fusion.GnssDeficitHandler
import com.example.idrnavigator.fusion.GnssState
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

import android.widget.Toast
import androidx.compose.material.icons.filled.Explore
import androidx.compose.material.icons.filled.GpsOff
import androidx.compose.material.icons.filled.Layers
import androidx.compose.material.icons.filled.MyLocation
import androidx.compose.material.icons.filled.Navigation
import androidx.compose.ui.platform.LocalContext
import com.example.idrnavigator.map.AppMapStyle
import com.example.idrnavigator.map.OsmMapController

@Composable
fun NavigationHudScreen(
    viewModel: NavigationViewModel,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val state by viewModel.uiState.collectAsState(initial = NavigationUiState())
    var currentMapView by remember { mutableStateOf<org.osmdroid.views.MapView?>(null) }
    var mapStyle by remember { mutableStateOf(AppMapStyle.DARK_COCKPIT) }
    var followVehicle by remember { mutableStateOf(true) }

    // Mode-duration counter: tracks how long current GNSS state has been active
    var lastTrackedState by remember { mutableStateOf(state.gnssState) }
    var stateStartTimeMs by remember { mutableLongStateOf(System.currentTimeMillis()) }
    var currentClockMs by remember { mutableLongStateOf(System.currentTimeMillis()) }

    LaunchedEffect(state.gnssState) {
        if (state.gnssState != lastTrackedState) {
            lastTrackedState = state.gnssState
            stateStartTimeMs = System.currentTimeMillis()
        }
    }

    LaunchedEffect(Unit) {
        while (true) {
            delay(500)
            currentClockMs = System.currentTimeMillis()
        }
    }

    val stateElapsedSec = ((currentClockMs - stateStartTimeMs) / 1000L).coerceAtLeast(0L)
    val stateMinutes = stateElapsedSec / 60
    val stateSeconds = stateElapsedSec % 60
    val modeDurationStr = String.format(Locale.US, "%s: %02d:%02d", state.gnssState.name, stateMinutes, stateSeconds)

    val targetPillColor = when (state.gnssState) {
        GnssState.GNSS_ACTIVE   -> GnssActive       // #4FD8E8 Cyan
        GnssState.TRANSITIONING -> GnssDegraded     // #F5A623 Amber
        GnssState.INS_ONLY      -> InsDeadReckoning // #7C5CFF Violet
    }

    val animatedPillColor by animateColorAsState(
        targetValue = targetPillColor,
        animationSpec = tween(durationMillis = 400),
        label = "pillGnssColor"
    )

    val gnssLabel = when (state.gnssState) {
        GnssState.GNSS_ACTIVE   -> if (state.gpsAccuracy > 20f) "GNSS DEGRADED" else "GNSS LOCKED"
        GnssState.TRANSITIONING -> "TRANSITIONING"
        GnssState.INS_ONLY      -> "INS ONLY"
    }

    Box(
        modifier = modifier.fillMaxSize()
    ) {
        // The live Map
        OsmMapController(
            uiState = state,
            modifier = Modifier.fillMaxSize(),
            mapStyle = mapStyle,
            followVehicle = followVehicle,
            onUserPan = { followVehicle = false },
            onMapReady = { currentMapView = it }
        )

        // Top Floating Pill
        Row(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 32.dp)
                .background(
                    color = CockpitSurface.copy(alpha = 0.85f),
                    shape = RoundedCornerShape(24.dp)
                )
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(12.dp)
                    .background(animatedPillColor, CircleShape)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = gnssLabel,
                color = CockpitPrimaryText,
                style = MaterialTheme.typography.labelLarge
            )
            Spacer(modifier = Modifier.width(16.dp))
            Text(
                text = "🛰 ${state.satelliteCount} Sats",
                color = if (state.satelliteCount > 0) CockpitPrimaryText else CockpitSecondaryText,
                style = MaterialTheme.typography.labelMedium
            )
            Spacer(modifier = Modifier.width(16.dp))
            Text(
                text = SimpleDateFormat("HH:mm:ss", Locale.US).format(Date()),
                color = CockpitSecondaryText,
                style = MaterialTheme.typography.labelMedium
            )
        }

        // Mount Slip Warning Banner
        if (state.isMountSlipped) {
            Surface(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 56.dp)
                    .fillMaxWidth(0.85f),
                shape = RoundedCornerShape(12.dp),
                color = GnssDegraded.copy(alpha = 0.9f),
                tonalElevation = 4.dp
            ) {
                Text(
                    text = "⚠ Phone mount slipped — recalibrating...",
                    color = Color.White,
                    style = MaterialTheme.typography.labelMedium,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                )
            }
        }

        // GPS Simulation Active Banner
        if (state.isGpsSimulationActive) {
            Surface(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = if (state.isMountSlipped) 96.dp else 56.dp)
                    .fillMaxWidth(0.85f),
                shape = RoundedCornerShape(12.dp),
                color = InsDeadReckoning.copy(alpha = 0.9f),
                tonalElevation = 4.dp
            ) {
                Text(
                    text = "🛰 GPS OUTAGE SIMULATION ACTIVE",
                    color = Color.White,
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = androidx.compose.ui.text.font.FontWeight.Bold,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                )
            }
        }

        // Dead Reckoning Model Toggle Chip (AI TinyTCN vs Classical)
        Surface(
            onClick = {
                viewModel.toggleDeadReckoningMode()
                val newMode = if (state.deadReckoningMode == com.example.idrnavigator.fusion.DeadReckoningMode.AI_TCN) "Classical (Strapdown)" else "AI (TinyTCN ONNX)"
                Toast.makeText(context, "Switched to: $newMode", Toast.LENGTH_SHORT).show()
            },
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 80.dp),
            shape = RoundedCornerShape(16.dp),
            color = CockpitSurface.copy(alpha = 0.85f),
            tonalElevation = 2.dp
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .background(
                            if (state.deadReckoningMode == com.example.idrnavigator.fusion.DeadReckoningMode.AI_TCN) InsDeadReckoning else CockpitSecondaryText,
                            CircleShape
                        )
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = if (state.deadReckoningMode == com.example.idrnavigator.fusion.DeadReckoningMode.AI_TCN) {
                        "MODEL: AI TCN (${state.aiLatencyMs}ms)"
                    } else {
                        "MODEL: CLASSICAL"
                    },
                    color = CockpitPrimaryText,
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = androidx.compose.ui.text.font.FontWeight.Bold
                )
            }
        }

        // Side Floating Action Buttons
        Column(
            modifier = Modifier
                .align(Alignment.CenterEnd)
                .padding(end = 16.dp, bottom = 120.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Re-center / Follow Vehicle Button
            FloatingActionButton(
                onClick = {
                    followVehicle = true
                    currentMapView?.let { map ->
                        if (state.gpsData.lat != 0.0 && state.gpsData.lon != 0.0) {
                            map.controller.animateTo(org.osmdroid.util.GeoPoint(state.gpsData.lat, state.gpsData.lon))
                            map.controller.setZoom(17.5)
                        }
                    }
                },
                containerColor = CockpitSurface,
                contentColor = if (followVehicle) VehicleMarkerAccent else CockpitSecondaryText,
                modifier = Modifier.size(48.dp),
                shape = CircleShape
            ) {
                Icon(Icons.Default.MyLocation, contentDescription = "Center on Vehicle")
            }

            // Map Style Switcher (Dark Cockpit, Street Navigation, Satellite View)
            FloatingActionButton(
                onClick = {
                    mapStyle = when (mapStyle) {
                        AppMapStyle.DARK_COCKPIT -> AppMapStyle.STREET_MAP
                        AppMapStyle.STREET_MAP -> AppMapStyle.SATELLITE
                        AppMapStyle.SATELLITE -> AppMapStyle.DARK_COCKPIT
                    }
                    Toast.makeText(context, mapStyle.title, Toast.LENGTH_SHORT).show()
                },
                containerColor = CockpitSurface,
                contentColor = CockpitPrimaryText,
                modifier = Modifier.size(48.dp),
                shape = CircleShape
            ) {
                Icon(Icons.Default.Layers, contentDescription = "Switch Map Style")
            }

            // Recalibrate IMU Button
            FloatingActionButton(
                onClick = { viewModel.recalibrate() },
                containerColor = CockpitSurface,
                contentColor = if (state.isCalibrated) CockpitPrimaryText else GnssDegraded,
                modifier = Modifier.size(48.dp),
                shape = CircleShape
            ) {
                Icon(Icons.Default.Refresh, contentDescription = "Recalibrate")
            }

            // GPS Outage Simulation Toggle
            FloatingActionButton(
                onClick = {
                    viewModel.toggleGpsSimulation()
                    val msg = if (!state.isGpsSimulationActive) "GPS Outage Simulation ON" else "GPS Outage Simulation OFF"
                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                },
                containerColor = if (state.isGpsSimulationActive) InsDeadReckoning.copy(alpha = 0.8f) else CockpitSurface,
                contentColor = if (state.isGpsSimulationActive) Color.White else CockpitSecondaryText,
                modifier = Modifier.size(48.dp),
                shape = CircleShape
            ) {
                Icon(Icons.Default.GpsOff, contentDescription = "Simulate GPS Outage")
            }

            // Course-Up / North-Up Toggle
            FloatingActionButton(
                onClick = {
                    viewModel.toggleCourseUpMode()
                    val msg = if (!state.isCourseUpMode) "Course-Up Mode" else "North-Up Mode"
                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                },
                containerColor = CockpitSurface,
                contentColor = if (state.isCourseUpMode) VehicleMarkerAccent else CockpitSecondaryText,
                modifier = Modifier.size(48.dp),
                shape = CircleShape
            ) {
                Icon(
                    if (state.isCourseUpMode) Icons.Default.Navigation else Icons.Default.Explore,
                    contentDescription = "Toggle Map Orientation"
                )
            }
        }

        // Bottom Instrument Panel
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(16.dp)
                .background(CockpitSurface, RoundedCornerShape(16.dp))
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            InstrumentDataBlock(
                label = "Speed",
                value = "%.0f".format(state.speedKmh),
                unit = "km/h"
            )
            Box(
                modifier = Modifier
                    .height(48.dp)
                    .width(1.dp)
                    .background(CockpitDivider)
            )
            // Drift color shifts amber→red as INS_ONLY duration increases
            val driftColor = when {
                state.gnssStatus != GnssStatus.INS_ONLY -> CockpitPrimaryText
                state.insOnlyDurationSec >= GnssDeficitHandler.INS_DRIFT_WARNING_THRESHOLD_SEC -> DriftCriticalRed
                state.insOnlyDurationSec >= GnssDeficitHandler.INS_DRIFT_WARNING_THRESHOLD_SEC * 0.67f -> DriftWarningAmber
                else -> CockpitPrimaryText
            }
            InstrumentDataBlock(
                label = if (state.gnssStatus == GnssStatus.INS_ONLY && state.insOnlyDurationSec >= GnssDeficitHandler.INS_DRIFT_WARNING_THRESHOLD_SEC)
                    "⚠ Drift Est." else "Drift Est.",
                value = if (state.hasFix || state.gnssStatus == GnssStatus.INS_ONLY) "%.1f".format(state.driftEstMeters) else "—",
                unit = "m",
                valueColor = driftColor,
                subText = modeDurationStr
            )
            Box(
                modifier = Modifier
                    .height(48.dp)
                    .width(1.dp)
                    .background(CockpitDivider)
            )
            InstrumentDataBlock(
                label = "Heading",
                value = "${state.headingDeg}",
                unit = "° ${state.headingCardinal}"
            )
        }
    }
}

@Composable
fun InstrumentDataBlock(
    label: String,
    value: String,
    unit: String,
    valueColor: Color = CockpitPrimaryText,
    subText: String? = null
) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = label,
            color = CockpitSecondaryText,
            style = MaterialTheme.typography.labelMedium
        )
        Row(verticalAlignment = Alignment.Bottom) {
            Text(
                text = value,
                color = valueColor,
                style = MaterialTheme.typography.displayMedium
            )
            Spacer(modifier = Modifier.width(4.dp))
            Text(
                text = unit,
                color = CockpitSecondaryText,
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.padding(bottom = 6.dp)
            )
        }
        if (subText != null) {
            Text(
                text = subText,
                color = valueColor.copy(alpha = 0.85f),
                style = MaterialTheme.typography.labelSmall
            )
        }
    }
}
