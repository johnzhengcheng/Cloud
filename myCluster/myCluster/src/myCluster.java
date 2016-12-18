import java.net.*;
//import java.lang.*;
//import java.io.*;
//import java.util.*;

public class myCluster {
	public int counter = 0;

	public static void main(String[] arstring) {

		dataChange pointer = new dataChange();
		pointer.setValue(1);

		System.out.println(pointer.flag);

		subThread mySubThread = new subThread(pointer);

		cmdThread mycmdThread= new cmdThread();
		
		mySubThread.start();

		try {
			while (pointer.flag > 0) {
				System.out.println("waiting");
				Thread.sleep(3000);

			}
			
	     mycmdThread.start();
			
			byte[] arb = new byte[] { 'h', 'e', 'l', 'l', 'o' };
			InetAddress inetAddress = InetAddress.getByName("230.0.0.1");
			DatagramPacket datagramPacket = new DatagramPacket(arb, arb.length,
					inetAddress, 7777);
			MulticastSocket multicastSocket = new MulticastSocket();
			while (true) {
				// Create a datagram package and send it to the multicast
				// group at 230.0.0.1, port 7777.
				// System.out.println("send");
				Thread.sleep(1000);
				multicastSocket.send(datagramPacket);
				Thread.sleep(1000);
			}

		} catch (Exception exception) {
			exception.printStackTrace();
		}

	}

}
