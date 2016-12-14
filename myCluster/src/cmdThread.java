import java.util.*;
import java.io.*;
import java.net.*;
import java.sql.*; //Java基础包，包含各种标准数据结构操作

public class cmdThread extends Thread {
	  
	  dataChange link;
	  	public cmdThread()
	  {
	        	
	  }
    
 
    public void run() {
      
    	try
        {
    		String cmd = "sh /root/myCluster/cluster.sh";
			Process child = Runtime.getRuntime().exec(cmd);
			System.out.println(child.exitValue());
			
			
			
			InputStream child_in = child.getInputStream();
			int c;
			while ((c = child_in.read()) != -1) {
				// System.out.println("kkk");
				System.out.print((char) c);
			}
			child_in.close();

			System.out.println("The ownership");

        }
        catch (Exception exception)
        {
        
        }
      }

      
    

}



